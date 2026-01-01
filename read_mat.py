import scipy.io as sio
import numpy as np

# 替换为你的2018.mat文件路径
mat_path = "/home/tjq/download/datasets/BSDS500/test/gt_mat/2018.mat"

# 读取.mat文件
mat = sio.loadmat(mat_path)

# 1. 打印文件的顶层键值
print("=== 顶层键值 ===")
for k in mat.keys():
    print(f"键名: {k}, 类型: {type(mat[k])}, 形状: {mat[k].shape if hasattr(mat[k], 'shape') else '无'}")

# 2. 解析groundTruth字段（核心标注）
print("\n=== groundTruth字段解析 ===")
gt = mat['groundTruth']
print(f"groundTruth类型: {type(gt)}, 形状: {gt.shape}, 数据类型: {gt.dtype}")

# 3. 解析第一个标注（BSDS500每张图有5个人工标注）
first_gt = gt[0, 0]  # 注意：mat文件的数组是(1,N)形状，需用[0,0]索引
print(f"\n第一个标注的字段: {first_gt.dtype.names}")

# 4. 解析Boundaries（边缘标注核心）
if 'Boundaries' in first_gt.dtype.names:
    boundaries = first_gt['Boundaries']
    print(f"Boundaries类型: {type(boundaries)}, 形状: {boundaries.shape}")
    # 提取第一个边缘掩码
    edge = boundaries[0, 0]
    print(f"单个边缘掩码: 类型={type(edge)}, 形状={edge.shape}, 数据类型={edge.dtype}, 取值范围={edge.min()}~{edge.max()}")
elif 'boundary' in first_gt.dtype.names:
    boundary = first_gt['boundary']
    print(f"boundary类型: {type(boundary)}, 形状: {boundary.shape}")
    edge = boundary[0, 0]
    print(f"单个边缘掩码: 类型={type(edge)}, 形状={edge.shape}, 数据类型={edge.dtype}, 取值范围={edge.min()}~{edge.max()}")

# 5. 验证所有5个标注的结构
print("\n=== 5个人工标注的边缘形状 ===")
for i in range(len(gt[0])):
    anno = gt[0, i]
    if 'Boundaries' in anno.dtype.names:
        edge = anno['Boundaries'][0, 0]
    else:
        edge = anno['boundary'][0, 0]
    print(f"标注{i+1}: 形状={edge.shape}, 非零像素数={np.count_nonzero(edge)}")