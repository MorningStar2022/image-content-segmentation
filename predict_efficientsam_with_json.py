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
import zipfile

# -------------------------- 导入模型相关模块模块 --------------------------
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from Hi_SAM.hi_sam.modeling.build import model_registry
from Hi_SAM.hi_sam.modeling.predictor import SamPredictor
from torchvision import transforms
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms
warnings.filterwarnings("ignore")


# -------------------------- COCO格式工具函数 --------------------------
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
    """将二值掩码转换为COCO RLE格式"""
    mask = mask.astype(np.uint8)
    rle = {"counts": [], "size": list(mask.shape)}
    counts = []
    prev = 0

    for pixel in mask.flatten(order='F'):
        if pixel != prev:
            counts.append(1)
            prev = pixel
        else:
            if counts:
                counts[-1] += 1
            else:
                counts.append(1)

    if not counts:
        counts = [mask.size]

    rle["counts"] = counts
    return rle


def add_coco_annotation(coco_data, img_id, mask, category_id):
    """向COCO数据中添加标注"""
    if np.sum(mask) == 0:
        print(f"⚠️ 跳过空掩码标注（类别ID: {category_id}，图像ID: {img_id}）")
        return

    area = int(np.sum(mask))
    where = np.argwhere(mask)
    if len(where) == 0:
        print(f"⚠️ 掩码无有效像素（类别ID: {category_id}，图像ID: {img_id}）")
        return

    y1, x1 = where.min(axis=0)
    y2, x2 = where.max(axis=0)
    bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]
    rle = mask_to_coco_rle(mask)

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


# -------------------------- Efficient-SAM工具函数 --------------------------
def process_small_region(rles):
    """处理小区域掩码"""
    new_masks = []
    scores = []
    min_area = 100
    nms_thresh = 0.7
    for rle in rles:
        mask = rle_to_mask(rle[0])

        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        scores.append(float(unchanged))

    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros_like(boxes[:, 0]),
        iou_threshold=nms_thresh,
    )

    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            rles[i_mask] = mask_to_rle_pytorch(mask_torch)
    masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
    return masks


def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):
    """获取给定嵌入和查询的预测结果"""
    predicted_masks, predicted_iou = model(
        img[None, ...], points, point_labels
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_masks = torch.take_along_dim(
        predicted_masks, sorted_ids[..., None, None], dim=2
    )
    predicted_masks = predicted_masks[0]
    iou = predicted_iou_scores[0, :, 0]
    index_iou = iou > 0.7
    iou_ = iou[index_iou]
    masks = predicted_masks[index_iou]
    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.9
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]
    masks = torch.ge(masks, 0.0)
    return masks, iou_


def run_everything_ours(img_path, model, grid_size=10):
    """运行Efficient-SAM的全图分割"""
    model = model.cpu()
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(image)
    _, original_image_h, original_image_w = img_tensor.shape

    # 生成网格点
    xy = []
    for i in range(grid_size):
        curr_x = 0.5 + i / grid_size * original_image_w
        for j in range(grid_size):
            curr_y = 0.5 + j / grid_size * original_image_h
            xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy))
    points = xy
    num_pts = xy.shape[0]
    point_labels = torch.ones(num_pts, 1)

    with torch.no_grad():
        predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
            img_tensor.cpu(),
            points.reshape(1, num_pts, 1, 2).cpu(),
            point_labels.reshape(1, num_pts, 1).cpu(),
            model.cpu(),
        )

    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    predicted_masks = process_small_region(rle)
    return predicted_masks, image.shape[:2]


# -------------------------- 全局配置与工具函数 --------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('Efficient-SAM + Hi-SAM + 掩码优化 高效流程', add_help=False)
    # 通用配置
    parser.add_argument("--input", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output", type=str, default='./final_results', help="结果保存根目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")

    # Efficient-SAM配置
    parser.add_argument("--efficientsam_model_type", type=str, default="vitt", choices=["vitt", "vits"],
                        help="Efficient-SAM模型类型 (vitt/vits)")
    parser.add_argument("--efficientsam_checkpoint", type=str, required=True, help="Efficient-SAM权重路径")
    parser.add_argument("--grid_size", type=int, default=32, help="分割网格大小")

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


# -------------------------- 掩码优化优化函数 --------------------------
def refine_edge_mask(
        edge_mask: np.ndarray,
        text_mask: Optional[np.ndarray] = None,
        edge_white_value: int = 255,
        fill_black_value: int = 0,
        text_dilate_pixel: int = 20
) -> np.ndarray:
    """优化边缘掩码"""
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

    refined_edge_mask = edge_mask_bin.copy()

    if text_mask is not None:
        if len(text_mask.shape) == 3:
            text_mask_gray = cv2.cvtColor(text_mask, cv2.COLOR_BGR2GRAY)
        else:
            text_mask_gray = text_mask.copy()
        _, text_mask_bin = cv2.threshold(text_mask_gray, 1, 255, cv2.THRESH_BINARY)

        dilate_kernel = np.ones((text_dilate_pixel * 2 + 1, text_dilate_pixel * 2 + 1), np.uint8)
        text_mask_dilated = cv2.dilate(text_mask_bin, dilate_kernel, iterations=1)

        text_edge_overlap = np.logical_and(edge_mask_bin == edge_white_value, text_mask_dilated == 255)
        refined_edge_mask[text_edge_overlap] = fill_black_value

    return refined_edge_mask


# -------------------------- 模型推理函数 --------------------------
def run_effsam_inference(img_path, effsam_model, device, grid_size=10):
    """Efficient-SAM推理：返回边缘掩码、物体掩码列表 + 推理耗时（毫秒单位）"""
    try:
        start_time = time.time()
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        # 运行全图分割
        predicted_masks, _ = run_everything_ours(img_path, effsam_model, grid_size)

        # 生成边缘掩码和物体掩码列表
        edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        object_masks = []

        for mask in predicted_masks:
            mask_np = mask.astype(np.uint8) * 255
            object_masks.append(mask_np)

            # 提取边缘
            edges = cv2.Canny(mask_np, threshold1=50, threshold2=150)
            edge_mask = cv2.bitwise_or(edge_mask, edges)

        # 计算耗时
        effsam_infer_time = round((time.time() - start_time) * 1000, 1)

        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": Path(img_path).stem,
            "img_size": (img_h, img_w),
            "sam_edge_mask": edge_mask,
            "object_masks": object_masks,
            "sam_infer_time": effsam_infer_time
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "sam_infer_time": 0.0
        }


def run_hisam_inference(img_path, hisam_model, hier_det=False, patch_mode=False):
    """Hi-SAM推理：返回文本掩码数组 + 推理耗时（毫秒）"""
    try:
        start_time = time.time()

        predictor = SamPredictor(hisam_model)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = Path(img_path).stem

        if patch_mode:
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
                input_point = np.array([[125, 275]])
                input_label = np.ones(input_point.shape[0])
                mask, hr_mask, score, hr_score, hi_mask, hi_iou, word_mask = predictor.predict(
                    multimask_output=False,
                    hier_det=True,
                    point_coords=input_point,
                    point_labels=input_label,
                )
                text_mask = hr_mask[0].astype(np.uint8) * 255
            else:
                mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
                text_mask = hr_mask[0].astype(np.uint8) * 255

        hisam_infer_time = round((time.time() - start_time) * 1000, 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": img_name,
            "hisam_text_mask": text_mask,
            "hisam_infer_time": hisam_infer_time
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "hisam_infer_time": 0.0
        }


# -------------------------- 主函数 --------------------------
def main():
    args = get_args_parser()

    # 创建结果目录
    os.makedirs(args.output, exist_ok=True)
    print(f"📁 结果保存目录：{args.output}")

    # 加载模型
    print("\n🚀 加载模型...")
    # 加载Efficient-SAM模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.efficientsam_model_type == "vitt":
        effsam = build_efficient_sam_vitt(args.efficientsam_checkpoint)
    else:  # vits
        # 处理压缩文件
        with zipfile.ZipFile(args.efficientsam_checkpoint, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(args.efficientsam_checkpoint))
        effsam = build_efficient_sam_vits(args.efficientsam_checkpoint)
    effsam.to(device)
    effsam.eval()

    # 加载Hi-SAM模型
    hisam = model_registry[args.hisam_model_type](args)
    hisam.eval()
    hisam.to(device)
    print(f"✅ 模型加载完成，使用设备：{device}")

    # 获取输入图像列表
    input_images = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            img_path = os.path.join(args.input, fname)
            if cv2.haveImageReader(img_path):
                input_images.append(img_path)
    else:
        input_images = glob.glob(os.path.expanduser(args.input))

    assert len(input_images) > 0, "❌ 未找到有效有效输入图像"
    print(f"\n📸 待处理图像数量：{len(input_images)}")

    # 初始化时间统计变量
    total_sam_time = 0.0
    total_hisam_time = 0.0
    success_sam_count = 0
    success_hisam_count = 0
    time_stats = []

    # 串行运行推理
    print("\n⚡ 开始串行推理（Efficient-SAM + Hi-SAM）...")
    inference_results = {}
    success_count = 0

    for img_idx, img_path in enumerate(tqdm(input_images, desc="推理+优化进度")):
        img_name = Path(img_path).stem
        inference_results[img_name] = {}

        # 初始化COCO数据
        coco_data = init_coco_format()
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        coco_data["images"].append({
            "id": img_idx + 1,
            "width": img_w,
            "height": img_h,
            "file_name": os.path.basename(img_path)
        })

        # 运行Efficient-SAM推理
        sam_result = run_effsam_inference(
            img_path=img_path,
            effsam_model=effsam,
            device=device,
            grid_size=args.grid_size
        )
        inference_results[img_name]["sam"] = sam_result

        if sam_result["status"] == "success":
            total_sam_time += sam_result["sam_infer_time"]
            success_sam_count += 1

        # 运行Hi-SAM推理
        hisam_result = run_hisam_inference(
            img_path=img_path,
            hisam_model=hisam,
            hier_det=args.hisam_hier_det,
            patch_mode=args.hisam_patch_mode
        )
        inference_results[img_name]["hisam"] = hisam_result

        if hisam_result["status"] == "success":
            total_hisam_time += hisam_result["hisam_infer_time"]
            success_hisam_count += 1

        # 记录耗时
        time_stats.append({
            "img_name": img_name,
            "sam_time": sam_result["sam_infer_time"],
            "hisam_time": hisam_result["hisam_infer_time"],
            "sam_status": sam_result["status"],
            "hisam_status": hisam_result["status"]
        })

        # 掩码优化与保存
        if sam_result["status"] == "success" and hisam_result["status"] == "success":
            sam_edge_mask = sam_result["sam_edge_mask"]
            hisam_text_mask = hisam_result["hisam_text_mask"]
            object_masks = sam_result["object_masks"]
            img_h, img_w = sam_result["img_size"]

            # 保存文本掩码
            text_mask_path = os.path.join(args.output, f"{img_name}_hisam_text_mask.png")
            cv2.imwrite(text_mask_path, hisam_text_mask)
            text_mask_bin = (hisam_text_mask > 127).astype(np.uint8)
            add_coco_annotation(coco_data, img_idx + 1, text_mask_bin, 1)

            # 优化边缘掩码
            refined_edge_mask = refine_edge_mask(
                edge_mask=sam_edge_mask,
                text_mask=hisam_text_mask,
                edge_white_value=args.edge_white_value,
                fill_black_value=args.fill_black_value,
                text_dilate_pixel=args.text_dilate_pixel
            )
            refined_mask_path = os.path.join(args.output, f"{img_name}_refined_edge_mask.png")
            cv2.imwrite(refined_mask_path, refined_edge_mask)
            edge_mask_bin = (refined_edge_mask > 127).astype(np.uint8)
            add_coco_annotation(coco_data, img_idx + 1, edge_mask_bin, 2)

            # 处理物体掩码
            text_mask_dilated = cv2.dilate(
                text_mask_bin,
                np.ones((5, 5), np.uint8),
                iterations=1
            )
            exclude_mask = np.logical_or(text_mask_dilated, edge_mask_bin).astype(np.uint8)

            combined_object_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for mask in object_masks:
                mask_bin = (mask > 127).astype(np.uint8)
                combined_object_mask = np.logical_or(combined_object_mask, mask_bin).astype(np.uint8)

            combined_object_mask = np.logical_and(combined_object_mask, 1 - exclude_mask).astype(np.uint8)

            object_mask_path = os.path.join(args.output, f"{img_name}_object_mask.png")
            cv2.imwrite(object_mask_path, combined_object_mask * 255)
            add_coco_annotation(coco_data, img_idx + 1, combined_object_mask, 3)

            # 保存COCO JSON
            coco_json_path = os.path.join(args.output, f"{img_name}_coco_annotations.json")
            with open(coco_json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False, indent=2)

            inference_results[img_name]["refined_edge_mask_path"] = refined_mask_path
            inference_results[img_name]["hisam_text_mask_path"] = text_mask_path
            inference_results[img_name]["object_mask_path"] = object_mask_path
            inference_results[img_name]["coco_json_path"] = coco_json_path
            success_count += 1
        else:
            print(f"\n⚠️ 跳过{img_name}：Efficient-SAM/Hi-SAM推理失败")
            if sam_result["status"] == "failed":
                print(f"   - Efficient-SAM失败原因：{sam_result['error']}")
            if hisam_result["status"] == "failed":
                print(f"   - Hi-SAM失败原因：{hisam_result['error']}")

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 时间统计结果
    print("\n" + "-" * 60)
    print("📊 推理时间统计（单位：毫秒 ms）")
    print("-" * 60)
    print(f"总处理图片数：{len(input_images)}")
    print(f"Efficient-SAM成功推理数：{success_sam_count} | Hi-SAM成功推理数：{success_hisam_count}")
    print(f"Efficient-SAM总耗时：{total_sam_time:.1f} ms | 平均每张：{total_sam_time / max(success_sam_count, 1):.1f} ms")
    print(f"Hi-SAM总耗时：{total_hisam_time:.1f} ms | 平均每张：{total_hisam_time / max(success_hisam_count, 1):.1f} ms")

    print("\n📋 单张图片耗时明细：")
    for stat in time_stats:
        status = f"Efficient-SAM: {stat['sam_status']} | Hi-SAM: {stat['hisam_status']}"
        print(
            f"  {stat['img_name']} | Efficient-SAM: {stat['sam_time']:.1f}ms | Hi-SAM: {stat['hisam_time']:.1f}ms | {status}")

    # 最终结果
    print("\n🎉 任务完成！成功处理 " + f"{success_count}/{len(input_images)} 张图像")
    print(f"📁 结果保存目录：{args.output}")
    print("📄 保存文件包括：")
    print("   - {img_name}_hisam_text_mask.png: Hi-SAM文本掩码")
    print("   - {img_name}_refined_edge_mask.png: 优化后的Efficient-SAM边缘掩码")
    print("   - {img_name}_object_mask.png: 物体掩码（排除文本和边缘）")
    print("   - {img_name}_coco_annotations.json: COCO格式标注文件（包含text/edge/object三类别）")


if __name__ == '__main__':
    main()