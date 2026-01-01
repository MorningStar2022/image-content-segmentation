import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation  # 邻域匹配核心
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

# ===================== 配置参数（对齐BSDS500官方）=====================
GT_DIR = "/home/tjq/download/datasets/BSDS500/test/gt_jpg"  # 真值边缘目录
PRED_DIR = "/home/tjq/download/datasets/BSDS500/test/fastsam_predict"  # 已做Canny的预测边缘
IMG_EXT = [".jpg", ".png", ".jpeg"]
TARGET_SIZE = (640,640)
DISTANCE_THRESHOLD = 1  # 3×3邻域
IOU_THRESHOLD = 0.5  # AP50阈值


# ===================== 核心工具函数（适配已做Canny的预测结果）=====================
def edge_matching(gt_edge, pred_edge, distance=DISTANCE_THRESHOLD):
    """
    param gt_edge: 真值边缘二值掩码 (H,W)
    param pred_edge: 已做Canny的预测边缘二值掩码 (H,W)
    return: TP, FP, FN
    """
    # 对真值边缘做膨胀
    gt_dilated = binary_dilation(gt_edge, structure=np.ones((2 * distance + 1, 2 * distance + 1)))

    # 计算TP/FP/FN
    TP = np.logical_and(pred_edge, gt_dilated).sum()  # 预测边缘在真值邻域内=正确
    FP = pred_edge.sum() - TP  # 预测边缘无真值匹配=误检
    FN = gt_edge.sum() - TP  # 真值边缘无预测匹配=漏检

    return TP, FP, FN


def load_edge_mask(path, target_size=TARGET_SIZE, is_gt=True):
    """
    加载边缘掩码（
    :param is_gt: True=真值边缘，False=已做Canny的预测边缘
    """
    # 加载图像并转为灰度图
    img = Image.open(path).convert("L")
    # 调整尺寸
    img = img.resize(target_size[::-1], Image.NEAREST)  # resize(W, H)
    img_arr = np.array(img)

    # 统一二值化（>0为边缘，0为非边缘）
    edge_mask = (img_arr > 0).astype(np.uint8)

    return edge_mask


def calculate_bsds_official_metrics(gt_dir, pred_dir):
    """
    邻域匹配
    """
    # 1. 匹配文件名（按无后缀名对齐）
    gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    pred_files = {os.path.splitext(f)[0]: f for f in os.listdir(pred_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    common_names = set(gt_files.keys()) & set(pred_files.keys())

    if len(common_names) == 0:
        raise ValueError("无匹配的真值/预测文件！检查文件名是否一致")
    print(f"匹配到 {len(common_names)} 张图像")

    # 2. 初始化存储变量
    # 全局PR曲线
    all_pred_probs = []
    all_gt_edges = []
    # 单图指标用
    pa_list = []
    fg_iou_list = []
    f1_list = []

    # 3. 逐图处理
    for name in tqdm(common_names, desc="计算指标"):
        # 构造路径
        gt_path = os.path.join(gt_dir, gt_files[name])
        pred_path = os.path.join(pred_dir, pred_files[name])

        # 加载边缘掩码（真值+已做Canny的预测）
        try:
            gt_edge = load_edge_mask(gt_path, is_gt=True)
            pred_edge = load_edge_mask(pred_path, is_gt=False)
        except Exception as e:
            print(f"跳过文件 {name}：{e}")
            continue

        # 存储像素级数据
        all_pred_probs.append(pred_edge.flatten().astype(np.float32))  # 预测概率（二值则为0/1）
        all_gt_edges.append(gt_edge.flatten())

        # 4. 邻域匹配计算TP/FP/FN
        TP, FP, FN = edge_matching(gt_edge, pred_edge)
        total_pixels = gt_edge.size
        TN = total_pixels - (TP + FP + FN)  # 背景匹配像素

        # 5. 计算单图指标
        # 5.1 PA
        PA = (TP + TN) / total_pixels if total_pixels > 0 else 0.0
        pa_list.append(PA)

        # 5.2 fgIoU
        fg_inter = TP
        fg_union = TP + FP + FN
        fg_iou = fg_inter / fg_union if fg_union > 0 else 0.0
        fg_iou_list.append(fg_iou)

        # 5.3 F1-Score
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_list.append(f1)

    # 合并所有像素的预测概率和真值
    all_pred_probs = np.concatenate(all_pred_probs)
    all_gt_edges = np.concatenate(all_gt_edges)
    # 计算全局PR曲线
    precision, recall, thresholds = precision_recall_curve(all_gt_edges, all_pred_probs)
    # 计算所有阈值的F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    ODS_F1 = np.max(f1_scores[:-1])
    # 计算AP50
    ap50 = 0.0
    recall_levels = np.linspace(0, 1, 11)
    for r in recall_levels:
        mask = recall >= r
        if np.any(mask):
            ap50 += np.max(precision[mask]) / 11.0

    # 8. 统计最终结果
    results = {
        "平均PA": np.mean(pa_list) if pa_list else 0.0,
        "平均fgIoU": np.mean(fg_iou_list) if fg_iou_list else 0.0,
        "平均F1-Score": np.mean(f1_list) if f1_list else 0.0,
        "AP50": ap50
    }
    return results


# ===================== 运行计算 =====================
if __name__ == "__main__":
    # 执行指标计算
    metrics_results = calculate_bsds_official_metrics(GT_DIR, PRED_DIR)

    # 打印结果
    print("=" * 60)
    print("BSDS500边缘检测指标")
    print("=" * 60)
    for metric, value in metrics_results.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 60)