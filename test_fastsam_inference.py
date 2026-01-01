import argparse
import glob
import time
from pathlib import Path

from tqdm import tqdm

import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
import cv2
import numpy as np
import os
from FastSAM.utils.tools import convert_box_xywh_to_xyxy
from ultralytics import YOLO
def run_fastsam_inference(img_path, fastsam_model, device, imgsz=1024, conf=0.4, iou=0.9):
    """Fast-SAM推理：返回边缘掩码数组、原始物体掩码列表，推理耗时"""
    try:
        start_time = time.time()
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        everything_results = fastsam_model(
            img_path,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )
        new_masks = margen(everything_results[0])
        new_masks = new_masks.astype(np.uint8)
        new_masks = new_masks * 255

        prompt_process = FastSAMPrompt(img_path, everything_results, device=device)
        ann = prompt_process.everything_prompt()


        edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        object_masks = []

        for mask in ann:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            object_masks.append(mask_np)
            edges = cv2.Canny(mask_np, threshold1=50, threshold2=150)
            edge_mask = cv2.bitwise_or(edge_mask, edges)

        fastsam_infer_time = round((time.time() - start_time) * 1000, 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "img_name": Path(img_path).stem,
            "img_size": (img_h, img_w),
            "sam_edge_mask": edge_mask,
            "object_masks": object_masks,
            "sam_infer_time": fastsam_infer_time,
            "new_masks": new_masks
        }
    except Exception as e:
        return {
            "status": "failed",
            "img_path": img_path,
            "error": str(e),
            "sam_infer_time": 0.0
        }

def margen(masks): # reuslts[0]

    conf = masks.boxes.conf     # mask的置信度
    mask = masks.masks[0]       # 用来获得原始图片的shape
    masks = masks.masks         # 所有mask
    final_mask = []
    for _, i in enumerate(masks):
        # 取mask
        mask = i.data[0].cpu().numpy()
        mask = mask.astype(np.uint8)
        x = cv2.Sobel(mask, cv2.CV_16S, 1, 0, ksize=3)
        y = cv2.Sobel(mask, cv2.CV_16S, 0, 1, ksize=3)
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        result = result >= 1
        # 将mask cat到final_mask 第一个维度上
        result = result * float(conf[_])
        final_mask.append(result)

    final_mask = np.array(final_mask)
    final_mask = np.amax(final_mask, axis=0)
    final_mask[0:4,:] = 0
    final_mask[-4:,:] = 0
    final_mask[:,0:4] = 0
    final_mask[:,-4:] = 0
    condition = (final_mask < 0.05) & (final_mask > 0.001)
    final_mask[condition] = 0.05
    return final_mask


def parse_args():
    # 通用配置
    parser = argparse.ArgumentParser('Fast-SAM', add_help=False)
    parser.add_argument("--input", type=str, default="/home/tjq/download/datasets/BSDS500/test/images", help="输入图像文件夹路径")
    parser.add_argument("--output", type=str, default="/home/tjq/download/datasets/BSDS500/test/fastsam_predict", help="结果保存根目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")

    # Fast-SAM配置
    parser.add_argument("--fastsam_checkpoint", type=str, default="FastSAM/weights/FastSAM-x.pt",
                        help="Fast-SAM权重路径")
    parser.add_argument("--fastsam_conf", type=float, default=0.001, help="Fast-SAM置信度阈值")
    parser.add_argument("--fastsam_iou", type=float, default=0.7, help="Fast-SAM IoU阈值")
    parser.add_argument("--fastsam_imgsz", type=int, default=960, help="Fast-SAM输入图像尺寸")
    return parser.parse_args()
def main():
    args = parse_args()

    # 创建结果目录
    os.makedirs(args.output, exist_ok=True)
    print(f"结果保存目录：{args.output}")

    # 加载模型
    print("\n加载模型...")
    fastsam = FastSAM(args.fastsam_checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # model=YOLO("FastSAM/weights/FastSAM-x.pt")


    # 获取输入图像列表
    input_images = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            img_path = os.path.join(args.input, fname)
            if cv2.haveImageReader(img_path):
                input_images.append(img_path)
    else:
        input_images = glob.glob(os.path.expanduser(args.input))

    assert len(input_images) > 0, "❌ 未找到有效输入图像"
    print(f"\n待处理图像数量：{len(input_images)}")

    # 初始化时间统计变量
    total_sam_time = 0.0
    success_sam_count = 0

    time_stats = []

    # 串行运行推理
    print("\n开始推理（Fast-SAM分割）...")
    inference_results = {}
    success_count = 0

    for img_idx, img_path in enumerate(tqdm(input_images, desc="推理")):
        img_name = Path(img_path).stem

        # results = model(img_path, device='0', retina_masks=True, iou=0.7, conf=0.001, imgsz=960,max_det=1000)
        # masks = margen(results[0])
        # masks = masks.astype(np.uint8)
        # masks = masks * 255
        # # 保存成图片
        # cv2.imwrite(os.path.join(args.output, f"{img_name}.png"), masks)

        inference_results[img_name] = {}

        # 添加图像信息到COCO数据
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]


        # 执行Fast-SAM推理
        sam_result = run_fastsam_inference(
            img_path=img_path,
            fastsam_model=fastsam,
            device=device,
            imgsz=args.fastsam_imgsz,
            conf=args.fastsam_conf,
            iou=args.fastsam_iou
        )
        inference_results[img_name]["sam"] = sam_result

        if sam_result["status"] == "success":
            total_sam_time += sam_result["sam_infer_time"]
            success_sam_count += 1

        # 记录单张图片耗时
        time_stats.append({
            "img_name": img_name,
            "sam_time": sam_result["sam_infer_time"],
            "sam_status": sam_result["status"],
        })
        sam_edge_mask = sam_result["sam_edge_mask"]
        new_sam_edge_mask = sam_result["new_masks"]
        edge_mask_path = os.path.join(args.output, f"{img_name}.jpg")
        cv2.imwrite(edge_mask_path, sam_edge_mask)
        # cv2.imwrite(edge_mask_path, new_sam_edge_mask)


        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 时间统计结果输出
    print("\n" + "-" * 60)
    print("推理时间统计（单位：毫秒 ms）")
    print("-" * 60)
    print(f"总处理图片数：{len(input_images)}")
    print(f"Fast-SAM总耗时：{total_sam_time:.1f} ms | 平均每张：{total_sam_time / max(success_sam_count, 1):.1f} ms")


    # 单张图片明细
    print("\n单张图片耗时明细：")
    for stat in time_stats:
        status = f"Fast-SAM: {stat['sam_status']}"
        print(
            f"  {stat['img_name']} | Fast-SAM: {stat['sam_time']:.1f}ms ")

    # 最终结果输出
    print(f"结果保存目录：{args.output}")


if __name__ == '__main__':
    main()