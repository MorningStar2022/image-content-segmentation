import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

# ===================== 配置参数 =====================
GT_DIR = "/home/tjq/PycharmProjects/Hi-SAM-main/datasets/TotalText/groundtruth_pixel/Test"  # 真值掩码目录
PRED_DIR = "/home/tjq/PycharmProjects/Hi-SAM-main/datasets/TotalText/predict/Test"  # 预测掩码目录
IMG_EXT = [".png", ".jpg", ".jpeg"]  # TotalText掩码多为png格式
DISTANCE_THRESHOLD = 1  # 邻域匹配阈值（3×3）
SIGMA = 1.0  # 高斯模糊核（生成概率图）


# ===================== 核心工具函数 =====================
def load_mask(path, is_gt=False):
    """
    加载掩码（适配TotalText格式）：
    - 真值掩码：文本区域=255，背景=0 → 转为1/0
    - 预测掩码：文本区域=任意非零值，背景=0 → 转为1/0
    """
    try:
        # 加载为灰度图
        img = Image.open(path).convert("L")
        mask = np.array(img)

        # 真值掩码处理：255→1，0→0
        if is_gt:
            mask = (mask == 255).astype(np.uint8)
        # 预测掩码处理：非零→1，0→0
        else:
            mask = (mask > 0).astype(np.uint8)

        return mask
    except Exception as e:
        print(f"加载掩码失败 {path}：{str(e)[:80]}")
        return None


def edge_matching(fg_gt, fg_pred, distance=DISTANCE_THRESHOLD):
    """
    前景匹配（文本区域像素级匹配）：
    - fg_gt: 真值文本掩码（1=文本，0=背景）
    - fg_pred: 预测文本掩码（1=文本，0=背景）
    返回TP/FP/FN（文本区域的真阳性/假阳性/假阴性）
    """
    if fg_gt is None or fg_pred is None:
        return 0, 0, 0

    # 邻域膨胀匹配（兼容小范围偏移）
    struct = np.ones((2 * distance + 1, 2 * distance + 1), dtype=np.uint8)
    gt_dilated = binary_dilation(fg_gt, structure=struct)

    # 计算TP/FP/FN
    TP = np.logical_and(fg_pred, gt_dilated).sum()  # 预测文本且匹配真值
    FP = fg_pred.sum() - TP  # 预测文本但无真值匹配
    FN = fg_gt.sum() - TP  # 真值文本但未预测

    return TP, FP, FN


def generate_prob_map(mask, sigma=SIGMA):
    """
    为预测掩码生成伪概率图（用于AP50计算）：
    - 膨胀生成软边缘 → 高斯模糊 → 归一化到0-1
    """
    if mask is None or np.sum(mask) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    # 多尺度膨胀生成概率梯度
    dilated_1 = binary_dilation(mask, structure=np.ones((3, 3)))
    dilated_2 = binary_dilation(mask, structure=np.ones((5, 5)))

    # 加权融合（核心区域概率高）
    prob_map = mask.astype(np.float32) * 1.0 + dilated_1.astype(np.float32) * 0.5 + dilated_2.astype(np.float32) * 0.2

    # 高斯模糊+归一化
    prob_map = cv2.GaussianBlur(prob_map, (7, 7), sigma)
    prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)

    return prob_map


def calculate_ap50(precision, recall):
    """
    计算AP50（11点插值法，TotalText通用评价标准）：
    - 遍历召回率0/0.1/.../1.0，取对应最大精确率求平均
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    ap50 = 0.0
    recall_levels = np.linspace(0, 1, 11)  # 11个召回率点

    for r in recall_levels:
        mask = recall >= r
        if np.any(mask):
            ap50 += np.max(precision[mask]) / 11.0

    return ap50


def calculate_f1_from_tp_fp_fn(TP, FP, FN):
    """从TP/FP/FN计算F1-Score"""
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


# ===================== 主计算流程 =====================
def calculate_totaltext_metrics(gt_dir, pred_dir):
    # 1. 匹配文件名（TotalText文件名一致，仅后缀不同）
    gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    pred_files = {os.path.splitext(f)[0]: f for f in os.listdir(pred_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    common_names = sorted(list(set(gt_files.keys()) & set(pred_files.keys())))

    if len(common_names) == 0:
        print("❌ 无匹配的真值/预测文件！")
        return {
            "平均PA": 0.0, "平均fgIoU": 0.0, "平均F1-Score": 0.0, "AP50": 0.0
        }

    print(f"✅ 匹配到 {len(common_names)} 张图像")

    # 2. 初始化存储变量
    all_pred_probs = []  # 所有像素的预测概率（用于AP50）
    all_gt_fg = []  # 所有像素的真值前景（用于AP50）
    pa_list = []  # 单图PA
    fg_iou_list = []  # 单图fgIoU
    f1_list = []  # 单图F1-Score
    valid_count = 0  # 有效处理图像数

    # 3. 逐图计算
    for name in tqdm(common_names, desc="计算TotalText指标"):
        # 构造路径
        gt_path = os.path.join(gt_dir, gt_files[name])
        pred_path = os.path.join(pred_dir, pred_files[name])

        # 加载真值和预测掩码
        gt_mask = load_mask(gt_path, is_gt=True)
        pred_mask = load_mask(pred_path, is_gt=False)

        if gt_mask is None or pred_mask is None:
            print(f"⚠️  跳过 {name}：掩码加载失败")
            continue

        # 尺寸校验（确保预测与真值尺寸一致）
        if gt_mask.shape != pred_mask.shape:
            # 强制resize预测掩码到真值尺寸
            pred_mask = cv2.resize(
                pred_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),  # (W, H)
                interpolation=cv2.INTER_NEAREST
            )
            print(f"⚠️  {name} 尺寸不匹配，已resize：{pred_mask.shape} → {gt_mask.shape}")

        # 生成预测概率图（用于AP50）
        pred_prob = generate_prob_map(pred_mask)
        if pred_prob is None:
            print(f"⚠️  跳过 {name}：概率图生成失败")
            continue

        # 存储全局数据（展平为1D数组）
        all_pred_probs.append(pred_prob.flatten())
        all_gt_fg.append(gt_mask.flatten())

        # 计算前景匹配（文本区域）
        TP, FP, FN = edge_matching(gt_mask, pred_mask)
        total_pixels = gt_mask.size
        TN = total_pixels - (TP + FP + FN)  # 背景正确像素

        # 计算单图指标
        # 1. PA（像素精度）
        PA = (TP + TN) / total_pixels if total_pixels > 0 else 0.0
        # 2. fgIoU（前景IoU，文本区域IoU）
        fg_iou = TP / (TP + FP + FN + 1e-8)
        # 3. F1-Score
        f1 = calculate_f1_from_tp_fp_fn(TP, FP, FN)

        # 存储指标
        pa_list.append(PA)
        fg_iou_list.append(fg_iou)
        f1_list.append(f1)
        valid_count += 1

    # 输出有效计数
    print(f"\n📊 有效处理图像数：{valid_count}/{len(common_names)}")
    if valid_count == 0:
        return {
            "平均PA": 0.0, "平均fgIoU": 0.0, "平均F1-Score": 0.0, "AP50": 0.0
        }

    # 4. 计算全局AP50
    concat_pred = np.concatenate(all_pred_probs)
    concat_gt = np.concatenate(all_gt_fg)
    precision, recall, _ = precision_recall_curve(concat_gt, concat_pred)
    ap50 = calculate_ap50(precision, recall)

    # 5. 汇总结果
    results = {
        "平均PA": np.mean(pa_list),
        "平均fgIoU": np.mean(fg_iou_list),
        "平均F1-Score": np.mean(f1_list),
        "AP50": ap50
    }

    return results


# ===================== 运行测试 =====================
if __name__ == "__main__":
    # 计算指标
    metrics = calculate_totaltext_metrics(GT_DIR, PRED_DIR)

    # 打印结果
    print("\n" + "=" * 60)
    print("TotalText文本分割指标")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("=" * 60)