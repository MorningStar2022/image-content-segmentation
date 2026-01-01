from ultralytics import YOLO
import numpy as np
from sklearn.metrics import f1_score

def calculate_segmentation_metrics(metrics):
    """
    从YOLO验证结果中提取掩码数据，计算fgIoU、PA、F-Score
    参数：
        metrics: model.val()返回的验证指标对象
    返回：
        seg_metrics: 包含fgIoU、PA、F1-Score的字典
    """
    # 初始化指标存储
    fg_iou_list = []
    pixel_accuracy_list = []
    f1_score_list = []

    # 遍历验证过程中的批次数据（需确保val时开启保存预测掩码，或通过自定义逻辑获取）
    # 方法1：从metrics中提取预测掩码和真实掩码（YOLOv11/v8通用）
    # 注意：需先在val时设置save=True，或通过自定义验证循环获取掩码
    val_loader = metrics.val_loader  # 获取验证数据加载器
    model = metrics.model  # 获取模型

    for batch in val_loader:
        # 前向推理获取预测掩码
        preds = model(batch['img'], verbose=False)
        for i, pred in enumerate(preds):
            # 获取单张图像的真实掩码（batch中第i张）
            true_masks = batch['masks'][i].cpu().numpy()  # 真实掩码: [num_obj, H, W]
            pred_masks = pred.masks.data.cpu().numpy()    # 预测掩码: [num_obj, H, W]

            # 处理单张图像的多目标掩码匹配（按IoU匹配真实与预测目标）
            if len(true_masks) == 0 or len(pred_masks) == 0:
                continue  # 无目标跳过

            # 1. 计算fgIoU（前景IoU，仅关注掩码区域，忽略背景）
            for true_mask, pred_mask in zip(true_masks, pred_masks):
                # 二值化掩码（确保只有0和1）
                true_mask_bin = (true_mask > 0).astype(np.uint8)
                pred_mask_bin = (pred_mask > 0).astype(np.uint8)

                # 计算前景交集和并集
                fg_intersection = np.logical_and(true_mask_bin, pred_mask_bin).sum()
                fg_union = np.logical_or(true_mask_bin, pred_mask_bin).sum()
                if fg_union == 0:
                    fg_iou = 0.0
                else:
                    fg_iou = fg_intersection / fg_union
                fg_iou_list.append(fg_iou)

                # 2. 计算PA（像素精度：正确分类的像素数 / 总像素数）
                total_pixels = true_mask_bin.size
                correct_pixels = (true_mask_bin == pred_mask_bin).sum()
                pa = correct_pixels / total_pixels
                pixel_accuracy_list.append(pa)

                # 3. 计算F-Score（像素级F1分数，基于前景像素）
                # 展平掩码为一维数组
                true_flat = true_mask_bin.flatten()
                pred_flat = pred_mask_bin.flatten()
                # 计算F1（默认binary模式，适用于二值掩码）
                f1 = f1_score(true_flat, pred_flat, average='binary', zero_division=0)
                f1_score_list.append(f1)

    # 计算所有目标的平均指标
    seg_metrics = {
        "fgIoU": np.mean(fg_iou_list) if fg_iou_list else 0.0,
        "PA": np.mean(pixel_accuracy_list) if pixel_accuracy_list else 0.0,
        "F1-Score": np.mean(f1_score_list) if f1_score_list else 0.0
    }
    return seg_metrics

# 主流程：加载模型+验证+计算自定义指标
if __name__ == "__main__":
    # 1. 加载模型
    model = YOLO("yolo_weights/yolo11m-seg.pt")  # 加载官方分割模型

    # 2. 验证模型（保留默认设置，可自定义参数如data、batch等）
    print("开始模型验证，计算mAP指标...")
    metrics = model.val()

    # 3. 打印默认mAP指标
    print("\n===== 默认mAP指标 =====")
    print(f"box_mAP50-95: {metrics.box.map:.4f}")
    print(f"box_mAP50: {metrics.box.map50:.4f}")
    print(f"seg_mAP50-95: {metrics.seg.map:.4f}")
    print(f"seg_mAP50: {metrics.seg.map50:.4f}")

    # 4. 计算并打印自定义指标（fgIoU/PA/F1-Score）
    print("\n===== 自定义分割指标 =====")
    custom_metrics = calculate_segmentation_metrics(metrics)
    print(f"平均fgIoU: {custom_metrics['fgIoU']:.4f}")
    print(f"平均像素精度(PA): {custom_metrics['PA']:.4f}")
    print(f"平均像素级F1-Score: {custom_metrics['F1-Score']:.4f}")