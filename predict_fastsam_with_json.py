import numpy as np
import torch
import warnings
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import os
import glob
import argparse
import sys
import time
from typing import Optional

# -------------------------- 导入模型相关模块 --------------------------
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from Hi_SAM.hi_sam.modeling.build import model_registry
from Hi_SAM.hi_sam.modeling.predictor import SamPredictor

warnings.filterwarnings("ignore")


# -------------------------- COCO格式工具函数 --------------------------
# -------------------------- COCO格式工具函数（修复版） --------------------------
def init_coco_format():
    """初始化COCO格式数据结构"""
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "text", "supercategory": "object"},
            {"id": 2, "name": "edge", "supercategory": "object"},
            {"id": 3, "name": "object", "supercategory": "object"}
        ],
        "images": [],
        "annotations": []
    }


def mask_to_coco_rle(mask):
    """将二值掩码转换为COCO RLE格式（修复全0掩码问题）"""
    mask = mask.astype(np.uint8)
    rle = {"counts": [], "size": list(mask.shape)}
    counts = []
    prev = 0

    # 遍历扁平化后的掩码（Fortran order，列优先）
    for pixel in mask.flatten(order='F'):
        if pixel != prev:
            # 像素值变化，新增计数项
            counts.append(1)
            prev = pixel
        else:
            # 像素值不变，累加计数（处理空列表情况）
            if counts:  # 仅当counts非空时累加
                counts[-1] += 1
            else:  # 空列表说明是第一个像素且为0，初始化计数
                counts.append(1)

    # 处理全0掩码的特殊情况（counts为空或仅一个元素）
    if not counts:
        counts = [mask.size]  # 全0时，计数为掩码总像素数

    rle["counts"] = counts
    return rle


def add_coco_annotation(coco_data, img_id, mask, category_id):
    """向COCO数据中添加标注（增强边界检查）"""
    # 1. 检查掩码是否全0，直接跳过
    if np.sum(mask) == 0:
        print(f"⚠️ 跳过空掩码标注（类别ID: {category_id}，图像ID: {img_id}）")
        return

    # 2. 计算掩码面积（0/1掩码的和即为面积）
    area = int(np.sum(mask))

    # 3. 计算边界框 (x, y, width, height)
    where = np.argwhere(mask)
    if len(where) == 0:
        print(f"⚠️ 掩码无有效像素（类别ID: {category_id}，图像ID: {img_id}）")
        return

    y1, x1 = where.min(axis=0)
    y2, x2 = where.max(axis=0)
    bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]

    # 4. 生成RLE编码
    rle = mask_to_coco_rle(mask)

    # 5. 创建标注（保证annotation ID唯一）
    annotation = {
        "id": len(coco_data["annotations"]) + 1,
        "image_id": img_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
        "attributes": {}
    }

    coco_data["annotations"].append(annotation)


# -------------------------- 全局配置与工具函数 --------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('Fast-SAM + Hi-SAM + 掩码优化 高效流程', add_help=False)
    # 通用配置
    parser.add_argument("--input", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output", type=str, default='./final_results', help="结果保存根目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")

    # Fast-SAM配置
    parser.add_argument("--fastsam_checkpoint", type=str, required=True, help="Fast-SAM权重路径")
    parser.add_argument("--fastsam_conf", type=float, default=0.4, help="Fast-SAM置信度阈值")
    parser.add_argument("--fastsam_iou", type=float, default=0.9, help="Fast-SAM IoU阈值")
    parser.add_argument("--fastsam_imgsz", type=int, default=640, help="Fast-SAM输入图像尺寸")

    # Hi-SAM配置
    parser.add_argument("--hisam_model_type", type=str, default="vit_l",
                        help="Hi-SAM模型类型 ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--hisam_checkpoint", type=str, required=True, help="Hi-SAM权重路径")
    parser.add_argument("--hisam_hier_det", action='store_true', help="Hi-SAM是否启用层级检测")
    parser.add_argument("--hisam_patch_mode", action='store_true', help="Hi-SAM是否启用patch模式")
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--attn_layers', default=1, type=int, help='cross attention layers数')
    parser.add_argument('--prompt_len', default=12, type=int, help='prompt token数')

    # 后处理配置
    parser.add_argument("--text_dilate_pixel", type=int, default=20, help="文本掩码膨胀像素数")
    parser.add_argument("--edge_white_value", type=int, default=255, help="边缘掩码白色值")
    parser.add_argument("--fill_black_value", type=int, default=0, help="重叠区域填充黑色值")

    return parser.parse_args()


# -------------------------- Hi-SAM工具函数 --------------------------
def patchify_sliding(image: np.array, patch_size: int = 512, stride: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j + patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i + patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])
    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]
    return whole_logits


# -------------------------- 掩码优化函数（无文件IO） --------------------------
def refine_edge_mask(
        edge_mask: np.ndarray,
        text_mask: Optional[np.ndarray] = None,
        edge_white_value: int = 255,
        fill_black_value: int = 0,
        text_dilate_pixel: int = 20
) -> np.ndarray:
    """优化SAM边缘掩码：纯内存操作，不涉及文件读写"""
    # 步骤1：统一边缘掩码为单通道二值格式
    if len(edge_mask.shape) == 3:
        edge_mask_gray = cv2.cvtColor(edge_mask, cv2.COLOR_BGR2GRAY)
    else:
        edge_mask_gray = edge_mask.copy()
    _, edge_mask_bin = cv2.threshold(
        edge_mask_gray,
        edge_white_value - 1,
        edge_white_value,
        cv2.THRESH_BINARY
    )

    # 步骤2：初始化优化后的边缘掩码
    refined_edge_mask = edge_mask_bin.copy()

    # 步骤3：处理文本掩码（核心）
    if text_mask is not None:
        # 文本掩码转单通道二值
        if len(text_mask.shape) == 3:
            text_mask_gray = cv2.cvtColor(text_mask, cv2.COLOR_BGR2GRAY)
        else:
            text_mask_gray = text_mask.copy()
        _, text_mask_bin = cv2.threshold(text_mask_gray, 1, 255, cv2.THRESH_BINARY)

        # 文本掩码膨胀
        dilate_kernel = np.ones((text_dilate_pixel * 2 + 1, text_dilate_pixel * 2 + 1), np.uint8)
        text_mask_dilated = cv2.dilate(text_mask_bin, dilate_kernel, iterations=1)

        # 重叠区域涂黑
        text_edge_overlap = np.logical_and(edge_mask_bin == edge_white_value, text_mask_dilated == 255)
        refined_edge_mask[text_edge_overlap] = fill_black_value

    return refined_edge_mask


# -------------------------- 模型推理函数（毫秒单位时间统计） --------------------------
def run_fastsam_inference(img_path, fastsam_model, device, imgsz=1024, conf=0.4, iou=0.9):
    """Fast-SAM推理：返回边缘掩码数组、原始物体掩码列表 + 推理耗时（毫秒）"""
    try:
        # 记录Fast-SAM推理开始时间
        start_time = time.time()

        # 读取图像
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        # 生成所有掩码（使用everything prompt）
        everything_results = fastsam_model(
            img_path,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )

        # 处理结果获取所有掩码
        prompt_process = FastSAMPrompt(img_path, everything_results, device=device)
        ann = prompt_process.everything_prompt()  # 获取所有掩码

        # 生成边缘掩码（仅内存操作）
        edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        object_masks = []  # 存储所有原始物体掩码

        for mask in ann:
            # 将掩码转换为二值图像
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            object_masks.append(mask_np)  # 保存原始物体掩码

            # 提取边缘
            edges = cv2.Canny(mask_np, threshold1=50, threshold2=150)
            # 合并到边缘掩码
            edge_mask = cv2.bitwise_or(edge_mask, edges)

        # 计算Fast-SAM推理耗时（转换为毫秒，保留1位小数）
        fastsam_infer_time = round((time.time() - start_time) * 1000, 1)

        # 清理显存缓存（缓解OOM）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": Path(img_path).stem,
            "img_size": (img_h, img_w),
            "sam_edge_mask": edge_mask,
            "object_masks": object_masks,  # 原始物体掩码列表
            "sam_infer_time": fastsam_infer_time  # 毫秒单位
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "sam_infer_time": 0.0  # 失败时耗时记为0
        }


def run_hisam_inference(img_path, hisam_model, hier_det=False, patch_mode=False):
    """Hi-SAM推理：返回文本掩码数组 + 推理耗时（毫秒）"""
    try:
        # 记录Hi-SAM推理开始时间
        start_time = time.time()

        predictor = SamPredictor(hisam_model)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = Path(img_path).stem

        if patch_mode:
            # Patch模式推理
            ori_size = image.shape[:2]
            patch_list, h_slice_list, w_slice_list = patchify_sliding(image_rgb, 512, 384)
            mask_512 = []
            for patch in patch_list:
                predictor.set_image(patch)
                m, hr_m, score, hr_score = predictor.predict(multimask_output=False, return_logits=True)
                mask_512.append(hr_m[0])
            mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
            text_mask = (mask_512 > predictor.model.mask_threshold).astype(np.uint8) * 255
        else:
            predictor.set_image(image_rgb)
            if hier_det:
                # 层级检测模式
                input_point = np.array([[125, 275]])
                input_label = np.ones(input_point.shape[0])
                mask, hr_mask, score, hr_score, hi_mask, hi_iou, word_mask = predictor.predict(
                    multimask_output=False,
                    hier_det=True,
                    point_coords=input_point,
                    point_labels=input_label,
                )
                text_mask = hr_mask[0].astype(np.uint8) * 255  # 转为0-255的单通道掩码
            else:
                # 普通文本分割模式
                mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
                text_mask = hr_mask[0].astype(np.uint8) * 255  # 转为0-255的单通道掩码

        # 计算Hi-SAM推理耗时（转换为毫秒，保留1位小数）
        hisam_infer_time = round((time.time() - start_time) * 1000, 1)

        # 清理显存缓存（缓解OOM）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": img_name,
            "hisam_text_mask": text_mask,
            "hisam_infer_time": hisam_infer_time  # 毫秒单位
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "hisam_infer_time": 0.0  # 失败时耗时记为0
        }


# -------------------------- 主函数（毫秒单位时间统计汇总） --------------------------
def main():
    args = get_args_parser()

    # 1. 创建结果目录
    os.makedirs(args.output, exist_ok=True)
    print(f"📁 结果保存目录：{args.output}")

    # 2. 加载模型
    print("\n🚀 加载模型...")
    # 加载Fast-SAM模型
    fastsam = FastSAM(args.fastsam_checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载Hi-SAM模型
    hisam = model_registry[args.hisam_model_type](args)
    hisam.eval()
    hisam.to(device)
    print(f"✅ 模型加载完成，使用设备：{device}")

    # 3. 获取输入图像列表
    input_images = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            img_path = os.path.join(args.input, fname)
            if cv2.haveImageReader(img_path):
                input_images.append(img_path)
    else:
        input_images = glob.glob(os.path.expanduser(args.input))

    assert len(input_images) > 0, "❌ 未找到有效输入图像"
    print(f"\n📸 待处理图像数量：{len(input_images)}")

    # 初始化时间统计变量（毫秒）
    total_sam_time = 0.0  # Fast-SAM总耗时（毫秒）
    total_hisam_time = 0.0  # Hi-SAM总耗时（毫秒）
    success_sam_count = 0  # Fast-SAM成功数
    success_hisam_count = 0  # Hi-SAM成功数
    time_stats = []  # 单张图片耗时明细

    # 4. 串行运行Fast-SAM + Hi-SAM推理
    print("\n⚡ 开始串行推理（Fast-SAM + Hi-SAM）...")
    inference_results = {}
    success_count = 0

    # 逐个处理每张图片
    for img_idx, img_path in enumerate(tqdm(input_images, desc="推理+优化进度")):
        img_name = Path(img_path).stem
        inference_results[img_name] = {}

        # 初始化COCO格式数据
        coco_data = init_coco_format()

        # 添加图像信息到COCO数据
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        coco_data["images"].append({
            "id": img_idx + 1,
            "width": img_w,
            "height": img_h,
            "file_name": os.path.basename(img_path)
        })

        # 4.1 串行执行Fast-SAM推理
        sam_result = run_fastsam_inference(
            img_path=img_path,
            fastsam_model=fastsam,
            device=device,
            imgsz=args.fastsam_imgsz,
            conf=args.fastsam_conf,
            iou=args.fastsam_iou
        )
        inference_results[img_name]["sam"] = sam_result

        # 累加Fast-SAM耗时（毫秒）
        if sam_result["status"] == "success":
            total_sam_time += sam_result["sam_infer_time"]
            success_sam_count += 1

        # 4.2 串行执行Hi-SAM推理
        hisam_result = run_hisam_inference(
            img_path=img_path,
            hisam_model=hisam,
            hier_det=args.hisam_hier_det,
            patch_mode=args.hisam_patch_mode
        )
        inference_results[img_name]["hisam"] = hisam_result

        # 累加Hi-SAM耗时（毫秒）
        if hisam_result["status"] == "success":
            total_hisam_time += hisam_result["hisam_infer_time"]
            success_hisam_count += 1

        # 记录单张图片耗时（毫秒）
        time_stats.append({
            "img_name": img_name,
            "sam_time": sam_result["sam_infer_time"],
            "hisam_time": hisam_result["hisam_infer_time"],
            "sam_status": sam_result["status"],
            "hisam_status": hisam_result["status"]
        })

        # 4.3 掩码优化 + 保存 + 生成COCO标注
        if sam_result["status"] == "success" and hisam_result["status"] == "success":
            # 直接从内存获取掩码数组
            sam_edge_mask = sam_result["sam_edge_mask"]
            hisam_text_mask = hisam_result["hisam_text_mask"]
            object_masks = sam_result["object_masks"]  # 获取原始物体掩码列表
            img_h, img_w = sam_result["img_size"]

            # 1. 保存文本掩码并添加到COCO
            text_mask_path = os.path.join(args.output, f"{img_name}_hisam_text_mask.png")
            cv2.imwrite(text_mask_path, hisam_text_mask)
            # 文本掩码二值化（0和1）
            text_mask_bin = (hisam_text_mask > 127).astype(np.uint8)
            add_coco_annotation(coco_data, img_idx + 1, text_mask_bin, 1)  # 类别1: text

            # 2. 内存中优化边缘掩码并添加到COCO
            refined_edge_mask = refine_edge_mask(
                edge_mask=sam_edge_mask,
                text_mask=hisam_text_mask,
                edge_white_value=args.edge_white_value,
                fill_black_value=args.fill_black_value,
                text_dilate_pixel=args.text_dilate_pixel
            )
            refined_mask_path = os.path.join(args.output, f"{img_name}_refined_edge_mask.png")
            cv2.imwrite(refined_mask_path, refined_edge_mask)
            # 边缘掩码二值化（0和1）
            edge_mask_bin = (refined_edge_mask > 127).astype(np.uint8)
            add_coco_annotation(coco_data, img_idx + 1, edge_mask_bin, 2)  # 类别2: edge

            # 3. 处理物体掩码（排除文本和边缘区域）并添加到COCO
            # 创建文本和边缘的联合掩码（需要排除的区域）
            text_mask_dilated = cv2.dilate(
                text_mask_bin,
                np.ones((5, 5), np.uint8),
                iterations=1
            )
            exclude_mask = np.logical_or(text_mask_dilated, edge_mask_bin).astype(np.uint8)

            # 合并所有物体掩码并排除文本和边缘区域
            combined_object_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for mask in object_masks:
                mask_bin = (mask > 127).astype(np.uint8)
                combined_object_mask = np.logical_or(combined_object_mask, mask_bin).astype(np.uint8)

                # 步骤2：创建排除掩码（文本+边缘，文本做轻微膨胀）
            text_mask_dilated = cv2.dilate(text_mask_bin, np.ones((5, 5), np.uint8), iterations=1)
            exclude_mask = np.logical_or(text_mask_dilated, edge_mask_bin).astype(np.uint8)

            # 步骤3：一次性排除文本和边缘区域
            combined_object_mask = np.logical_and(combined_object_mask, 1 - exclude_mask).astype(np.uint8)

            # 保存物体掩码
            object_mask_path = os.path.join(args.output, f"{img_name}_object_mask.png")
            cv2.imwrite(object_mask_path, combined_object_mask * 255)
            # 添加到COCO标注
            add_coco_annotation(coco_data, img_idx + 1, combined_object_mask, 3)  # 类别3: object

            # 4. 保存COCO格式JSON文件
            coco_json_path = os.path.join(args.output, f"{img_name}_coco_annotations.json")
            with open(coco_json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False, indent=2)

            inference_results[img_name]["refined_edge_mask_path"] = refined_mask_path
            inference_results[img_name]["hisam_text_mask_path"] = text_mask_path
            inference_results[img_name]["object_mask_path"] = object_mask_path
            inference_results[img_name]["coco_json_path"] = coco_json_path
            success_count += 1
        else:
            # 打印失败信息
            print(f"\n⚠️ 跳过{img_name}：Fast-SAM/Hi-SAM推理失败")
            if sam_result["status"] == "failed":
                print(f"   - Fast-SAM失败原因：{sam_result['error']}")
            if hisam_result["status"] == "failed":
                print(f"   - Hi-SAM失败原因：{hisam_result['error']}")

        # 强制清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 5. 时间统计结果输出（毫秒单位）
    print("\n" + "-" * 60)
    print("📊 推理时间统计（单位：毫秒 ms）")
    print("-" * 60)
    # 整体统计
    print(f"总处理图片数：{len(input_images)}")
    print(f"Fast-SAM成功推理数：{success_sam_count} | Hi-SAM成功推理数：{success_hisam_count}")
    print(f"Fast-SAM总耗时：{total_sam_time:.1f} ms | 平均每张：{total_sam_time / max(success_sam_count, 1):.1f} ms")
    print(f"Hi-SAM总耗时：{total_hisam_time:.1f} ms | 平均每张：{total_hisam_time / max(success_hisam_count, 1):.1f} ms")

    # 单张图片明细（毫秒单位）
    print("\n📋 单张图片耗时明细：")
    for stat in time_stats:
        status = f"Fast-SAM: {stat['sam_status']} | Hi-SAM: {stat['hisam_status']}"
        print(
            f"  {stat['img_name']} | Fast-SAM: {stat['sam_time']:.1f}ms | Hi-SAM: {stat['hisam_time']:.1f}ms | {status}")

    # 6. 最终结果输出
    print("\n🎉 任务完成！成功处理 "+f"{success_count}/{len(input_images)} 张图像")
    print(f"📁 结果保存目录：{args.output}")
    print("📄 保存文件包括：")
    print("   - {img_name}_hisam_text_mask.png: Hi-SAM文本掩码")
    print("   - {img_name}_refined_edge_mask.png: 优化后的Fast-SAM边缘掩码")
    print("   - {img_name}_object_mask.png: 物体掩码（排除文本和边缘）")
    print("   - {img_name}_coco_annotations.json: COCO格式标注文件（包含text/edge/object三类别）")


if __name__ == '__main__':
    main()