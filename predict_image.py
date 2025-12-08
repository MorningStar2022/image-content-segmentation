import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
import pyclipper
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
# 1. 获取当前脚本（final_combined.py）的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取项目根目录（image-seg）—— 即脚本所在目录
project_root = os.path.dirname(current_script_path)

# 3. 将SAM和Hi-SAM的源码目录加入Python路径（关键！）
# 加入SAM源码根目录（segment-anything-main）
sam_source_dir = os.path.join(project_root, "segment-anything-main")
if sam_source_dir not in sys.path:
    sys.path.insert(0, sam_source_dir)

# 加入Hi-SAM源码根目录（Hi-SAM-main）
hisam_source_dir = os.path.join(project_root, "Hi-SAM-main")
if hisam_source_dir not in sys.path:
    sys.path.insert(0, hisam_source_dir)
# -------------------------- 导入模型相关模块 --------------------------
# SAM相关

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Hi-SAM相关
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor

warnings.filterwarnings("ignore")


# -------------------------- 全局配置与工具函数 --------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('SAM + Hi-SAM + 掩码优化 完整流程', add_help=False)
    # 通用配置
    parser.add_argument("--input", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output", type=str, default='./final_results', help="结果保存根目录")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--max_workers", type=int, default=2, help="并行线程数（SAM/Hi-SAM）")

    # SAM配置
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM模型类型 ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM权重路径")
    parser.add_argument("--sam_max_masks", type=int, default=300, help="SAM最大掩码数")
    parser.add_argument("--sam_transparency", type=float, default=0.3, help="SAM可视化透明度")

    # Hi-SAM配置
    parser.add_argument("--hisam_model_type", type=str, default="vit_l",
                        help="Hi-SAM模型类型 ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--hisam_checkpoint", type=str, required=True, help="Hi-SAM权重路径")
    parser.add_argument("--hisam_hier_det", action='store_true', help="Hi-SAM是否启用层级检测")
    parser.add_argument("--hisam_patch_mode", action='store_true', help="Hi-SAM是否启用patch模式")

    # 掩码优化配置
    parser.add_argument("--text_dilate_pixel", type=int, default=20, help="文本掩码膨胀像素数")
    parser.add_argument("--edge_white_value", type=int, default=255, help="边缘掩码白色值")
    parser.add_argument("--fill_black_value", type=int, default=0, help="重叠区域填充黑色值")

    return parser.parse_args()


# -------------------------- SAM工具函数 --------------------------
def draw_segmentation(anns, max_masks=300):
    if len(anns) == 0:
        return
    h, w = anns[0]['segmentation'].shape
    image = np.zeros((h, w, 3), dtype=np.float64)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    no_masks = min(len(sorted_anns), max_masks)

    # 生成随机颜色
    colors = []
    for i in range(max_masks):
        colors.append(np.random.random((3)))

    for i in range(no_masks):
        seg = sorted_anns[i]['segmentation']
        image[seg] = colors[i]
    return image


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


def save_binary_mask(mask: np.array, filename):
    if len(mask.shape) == 3:
        assert mask.shape[0] == 1
        mask = mask[0].astype(np.uint8) * 255
    elif len(mask.shape) == 2:
        mask = mask.astype(np.uint8) * 255
    else:
        raise NotImplementedError
    mask = Image.fromarray(mask)
    mask.save(filename)


# -------------------------- 掩码优化函数（第三个脚本核心） --------------------------
def refine_edge_mask(
        edge_mask: np.ndarray,
        text_mask: Optional[np.ndarray] = None,
        image_mask: Optional[np.ndarray] = None,
        edge_white_value: int = 255,
        fill_black_value: int = 0,
        text_dilate_pixel: int = 20,
        save_refined_mask: bool = True,
        save_path: str = "refined_edge_mask.jpg"
) -> np.ndarray:
    """优化SAM边缘掩码：用Hi-SAM文本掩码涂黑重叠区域"""
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

    # 步骤4：处理图像掩码（可选）
    if image_mask is not None:
        if len(image_mask.shape) == 3:
            image_mask_gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
        else:
            image_mask_gray = image_mask.copy()
        _, image_mask_bin = cv2.threshold(image_mask_gray, 1, 255, cv2.THRESH_BINARY)
        image_edge_overlap = np.logical_and(edge_mask_bin == edge_white_value, image_mask_bin == 255)
        refined_edge_mask[image_edge_overlap] = fill_black_value

    # 步骤5：保存优化后的掩码
    if save_refined_mask:
        cv2.imwrite(save_path, refined_edge_mask)
        print(f"✅ 优化后的边缘掩码已保存：{save_path}")

    return refined_edge_mask


# -------------------------- 模型推理函数 --------------------------
def run_sam_inference(img_path, sam_model, output_dir, max_masks=300, transparency=0.3):
    """SAM推理：生成边缘掩码+可视化图，返回掩码路径"""
    try:
        mask_generator = SamAutomaticMaskGenerator(sam_model)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成SAM掩码
        masks = mask_generator.generate(image_rgb)

        # 1. 生成边缘掩码
        edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask_data in masks:
            mask = mask_data["segmentation"].astype(np.uint8) * 255
            edges = cv2.Canny(mask, threshold1=50, threshold2=150)
            edge_mask = cv2.bitwise_or(edge_mask, edges)

        # 保存SAM边缘掩码
        img_name = Path(img_path).stem
        sam_edge_mask_path = os.path.join(output_dir, f"{img_name}_sam_edge_mask.png")
        cv2.imwrite(sam_edge_mask_path, edge_mask)

        # 2. 生成可视化图
        image_float = image_rgb.astype(np.float64) / 255
        seg = draw_segmentation(masks, max_masks)
        if seg is not None:
            image_float += transparency * seg
        image_out = (255 * image_float).astype(np.uint8)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
        sam_vis_path = os.path.join(output_dir, f"{img_name}_sam_vis.png")
        cv2.imwrite(sam_vis_path, image_out)

        return {
            "status": "success",
            "img_name": img_name,
            "sam_edge_mask_path": sam_edge_mask_path,
            "sam_vis_path": sam_vis_path
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e)
        }


def run_hisam_inference(img_path, hisam_model, output_dir, hier_det=False, patch_mode=False):
    """Hi-SAM推理：生成文本掩码，返回掩码路径"""
    try:
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
            mask = mask_512 > predictor.model.mask_threshold
            hisam_mask_path = os.path.join(output_dir, f"{img_name}_hisam_text_mask.png")
            save_binary_mask(mask, hisam_mask_path)
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
                hisam_mask_path = os.path.join(output_dir, f"{img_name}_hisam_hier_mask.png")
                save_binary_mask(hr_mask, hisam_mask_path)
            else:
                # 普通文本分割模式
                mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
                hisam_mask_path = os.path.join(output_dir, f"{img_name}_hisam_text_mask.png")
                save_binary_mask(hr_mask, hisam_mask_path)

        return {
            "status": "success",
            "img_name": img_name,
            "hisam_text_mask_path": hisam_mask_path
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e)
        }


# -------------------------- 主流程：推理 + 掩码优化 --------------------------
def main():
    args = get_args_parser()

    # 1. 创建结果目录
    os.makedirs(args.output, exist_ok=True)
    print(f" 结果保存目录：{args.output}")

    # 2. 加载模型
    print("\n 加载模型...")
    # 加载SAM
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    # 加载Hi-SAM
    hisam = model_registry[args.hisam_model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print(" 模型加载完成")

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
    print(f"\n 待处理图像数量：{len(input_images)}")

    # 4. 并行运行SAM + Hi-SAM推理
    print("\n⚡ 开始并行推理（SAM + Hi-SAM）...")
    inference_results = {}  # 存储每张图的推理结果
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        # 提交任务
        for img_path in input_images:
            img_name = Path(img_path).stem
            # SAM任务
            fut_sam = executor.submit(
                run_sam_inference,
                img_path=img_path,
                sam_model=sam,
                output_dir=args.output,
                max_masks=args.sam_max_masks,
                transparency=args.sam_transparency
            )
            # Hi-SAM任务
            fut_hisam = executor.submit(
                run_hisam_inference,
                img_path=img_path,
                hisam_model=hisam,
                output_dir=args.output,
                hier_det=args.hisam_hier_det,
                patch_mode=args.hisam_patch_mode
            )
            futures[fut_sam] = ("sam", img_name)
            futures[fut_hisam] = ("hisam", img_name)

        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="推理进度"):
            task_type, img_name = futures[future]
            result = future.result()
            if img_name not in inference_results:
                inference_results[img_name] = {}
            if task_type == "sam":
                inference_results[img_name]["sam"] = result
            else:
                inference_results[img_name]["hisam"] = result

    # 5. 掩码优化：用Hi-SAM文本掩码优化SAM边缘掩码
    print("\n 开始优化SAM边缘掩码...")
    for img_name in tqdm(inference_results.keys(), desc="优化进度"):
        res = inference_results[img_name]
        # 跳过推理失败的图像
        if res.get("sam", {}).get("status") != "success" or res.get("hisam", {}).get("status") != "success":
            print(f"\n⚠ 跳过{img_name}：SAM/Hi-SAM推理失败")
            continue

        # 读取掩码
        sam_edge_mask = cv2.imread(res["sam"]["sam_edge_mask_path"], 0)
        hisam_text_mask = cv2.imread(res["hisam"]["hisam_text_mask_path"], 0)

        # 优化掩码
        refined_mask_path = os.path.join(args.output, f"{img_name}_refined_edge_mask.png")
        refine_edge_mask(
            edge_mask=sam_edge_mask,
            text_mask=hisam_text_mask,
            edge_white_value=args.edge_white_value,
            fill_black_value=args.fill_black_value,
            text_dilate_pixel=args.text_dilate_pixel,
            save_refined_mask=True,
            save_path=refined_mask_path
        )
        inference_results[img_name]["refined_edge_mask_path"] = refined_mask_path

    # 6. 输出最终结果汇总
    print("\n 处理结果汇总：")
    success_count = 0
    for img_name, res in inference_results.items():
        if res.get("sam", {}).get("status") == "success" and res.get("hisam", {}).get("status") == "success":
            success_count += 1
            print(f"✅ {img_name}：")
            print(f"   - SAM边缘掩码：{res['sam']['sam_edge_mask_path']}")
            print(f"   - Hi-SAM文本掩码：{res['hisam']['hisam_text_mask_path']}")
            print(f"   - 优化后边缘掩码：{res['refined_edge_mask_path']}")
        else:
            print(f"❌ {img_name}：处理失败")
            if res.get("sam", {}).get("status") == "failed":
                print(f"   - SAM失败原因：{res['sam']['error']}")
            if res.get("hisam", {}).get("status") == "failed":
                print(f"   - Hi-SAM失败原因：{res['hisam']['error']}")

    print(f"\n 任务完成！成功处理 {success_count}/{len(inference_results)} 张图像")
    print(f" 所有结果已保存至：{args.output}")


if __name__ == '__main__':
    main()