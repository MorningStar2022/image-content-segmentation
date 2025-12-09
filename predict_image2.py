import numpy as np
import torch
import warnings
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import os
import glob
import argparse
import sys
import time  # 导入时间模块
from typing import Optional

# -------------------------- 导包路径配置 --------------------------
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)

hisam_source_dir = os.path.join(project_root, "Hi-SAM-main")
if hisam_source_dir not in sys.path:
    sys.path.insert(0, hisam_source_dir)

sam_source_dir = os.path.join(project_root, "segment-anything-main")
if sam_source_dir not in sys.path:
    sys.path.insert(0, sam_source_dir)


# -------------------------- 导入模型相关模块 --------------------------
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor

warnings.filterwarnings("ignore")


# -------------------------- 全局配置与工具函数 --------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('SAM + Hi-SAM + 掩码优化 高效流程', add_help=False)
    # 通用配置
    parser.add_argument("--input", type=str, default="./input", help="输入图像文件夹路径")
    parser.add_argument("--output", type=str, default='./final_results', help="结果保存根目录")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")

    # SAM配置
    parser.add_argument("--sam_model_type", type=str, default="vit_b", help="SAM模型类型 ['vit_t']")
    parser.add_argument("--sam_checkpoint", type=str, default="segment-anything-main/pretrain_model/sam_vit_b_01ec64.pth", help="SAM权重路径")
    parser.add_argument("--sam_max_masks", type=int, default=300, help="SAM最大掩码数")

    # Hi-SAM配置
    parser.add_argument("--hisam_model_type", type=str, default="vit_s",
                        help="Hi-SAM模型类型 ['vit_h', 'vit_l', 'vit_b','vit_s']")
    parser.add_argument("--hisam_checkpoint", default="Hi-SAM-main/pretrained_checkpoint/efficient_hi_sam_s.pth",type=str,  help="Hi-SAM权重路径")
    parser.add_argument("--hisam_hier_det", default=False,action='store_true', help="Hi-SAM是否启用层级检测")
    parser.add_argument("--hisam_patch_mode", default=False,action='store_true', help="Hi-SAM是否启用patch模式")
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
def run_sam_inference(img_path, sam_model, max_masks=300):
    """SAM推理：返回边缘掩码数组 + 推理耗时（毫秒）"""
    try:
        # 记录SAM推理开始时间
        start_time = time.time()

        mask_generator = SamAutomaticMaskGenerator(sam_model)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成SAM掩码
        masks = mask_generator.generate(image_rgb)

        # 生成边缘掩码（仅内存操作）
        edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask_data in masks:
            mask = mask_data["segmentation"].astype(np.uint8) * 255
            edges = cv2.Canny(mask, threshold1=50, threshold2=150)
            edge_mask = cv2.bitwise_or(edge_mask, edges)

        # 计算SAM推理耗时（转换为毫秒，保留1位小数）
        sam_infer_time = round((time.time() - start_time) * 1000, 1)

        # 清理显存缓存（缓解OOM）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": Path(img_path).stem,
            "sam_edge_mask": edge_mask,
            "sam_infer_time": sam_infer_time  # 毫秒单位
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
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    sam.eval()
    hisam = model_registry[args.hisam_model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("✅ 模型加载完成")

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
    total_sam_time = 0.0  # SAM总耗时（毫秒）
    total_hisam_time = 0.0  # Hi-SAM总耗时（毫秒）
    success_sam_count = 0  # SAM成功数
    success_hisam_count = 0  # Hi-SAM成功数
    time_stats = []  # 单张图片耗时明细

    # 4. 串行运行SAM + Hi-SAM推理
    print("\n⚡ 开始串行推理（SAM + Hi-SAM）...")
    inference_results = {}
    success_count = 0

    # 逐个处理每张图片
    for img_path in tqdm(input_images, desc="推理+优化进度"):
        img_name = Path(img_path).stem
        inference_results[img_name] = {}

        # 4.1 串行执行SAM推理
        sam_result = run_sam_inference(
            img_path=img_path,
            sam_model=sam,
            max_masks=args.sam_max_masks
        )
        inference_results[img_name]["sam"] = sam_result

        # 累加SAM耗时（毫秒）
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

        # 4.3 掩码优化 + 保存
        if sam_result["status"] == "success" and hisam_result["status"] == "success":
            # 直接从内存获取掩码数组
            sam_edge_mask = sam_result["sam_edge_mask"]
            hisam_text_mask = hisam_result["hisam_text_mask"]

            # 1. 保存文本掩码
            text_mask_path = os.path.join(args.output, f"{img_name}_hisam_text_mask.png")
            cv2.imwrite(text_mask_path, hisam_text_mask)

            # 2. 内存中优化边缘掩码
            refined_edge_mask = refine_edge_mask(
                edge_mask=sam_edge_mask,
                text_mask=hisam_text_mask,
                edge_white_value=args.edge_white_value,
                fill_black_value=args.fill_black_value,
                text_dilate_pixel=args.text_dilate_pixel
            )

            # 3. 保存优化后的边缘掩码
            refined_mask_path = os.path.join(args.output, f"{img_name}_refined_edge_mask.png")
            cv2.imwrite(refined_mask_path, refined_edge_mask)

            inference_results[img_name]["refined_edge_mask_path"] = refined_mask_path
            inference_results[img_name]["hisam_text_mask_path"] = text_mask_path
            success_count += 1
        else:
            # 打印失败信息
            print(f"\n⚠️ 跳过{img_name}：SAM/Hi-SAM推理失败")
            if sam_result["status"] == "failed":
                print(f"   - SAM失败原因：{sam_result['error']}")
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
    print(f"SAM成功推理数：{success_sam_count} | Hi-SAM成功推理数：{success_hisam_count}")
    print(f"SAM总耗时：{total_sam_time:.1f} ms | 平均每张：{total_sam_time / max(success_sam_count, 1):.1f} ms")
    print(f"Hi-SAM总耗时：{total_hisam_time:.1f} ms | 平均每张：{total_hisam_time / max(success_hisam_count, 1):.1f} ms")

    # 单张图片明细（毫秒单位）
    print("\n📋 单张图片耗时明细：")
    for stat in time_stats:
        status = f"SAM: {stat['sam_status']} | Hi-SAM: {stat['hisam_status']}"
        print(f"  {stat['img_name']} | SAM: {stat['sam_time']:.1f}ms | Hi-SAM: {stat['hisam_time']:.1f}ms | {status}")

    # 6. 最终结果输出
    print("\n🎉 任务完成！成功处理 {success_count}/{len(input_images)} 张图像")
    print(f"📁 结果保存目录：{args.output}")
    print("📄 仅保存以下文件：")
    print("   - {img_name}_hisam_text_mask.png: Hi-SAM文本掩码")
    print("   - {img_name}_refined_edge_mask.png: 优化后的SAM边缘掩码")


if __name__ == '__main__':
    main()