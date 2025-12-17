import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from tqdm import tqdm
# 导入SAM相关模块
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#
#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     ax.imshow(img)
#
#
# image = cv2.imread('test/4_300_300c_2480_3512.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
#
# sam_checkpoint = "pretrain_model/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
#
# device = "cuda"
#
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
#
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)
#
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show()
# # 保存图片（关键修改）
# output_path = os.path.join("test/res", "segmentation_result.png")
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 去除多余边距


# config
in_dir = 'test/input/'
out_dir = 'test/res/'
sam_model = "vit_h"
sam_check = "pretrain_model/sam_vit_h_4b8939.pth"

device="cuda"
transparency = 0.3
max_masks = 300

# # sam generator params
# points_per_batch=64
# points_per_side=64
# pred_iou_thresh=0.86
# stability_score_thresh=0.92
# crop_n_layers=1
# crop_n_points_downscale_factor=2
# min_mask_region_area=100

# list of random colors
colors = []
for i in range(max_masks):
    colors.append(np.random.random((3)))


def draw_segmentation(anns):
    if len(anns) == 0:
        return
    h, w = anns[0]['segmentation'].shape
    image = np.zeros((h, w, 3), dtype=np.float64)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    no_masks = min(len(sorted_anns), max_masks)
    for i in range(no_masks):
        # true/false segmentation
        seg = sorted_anns[i]['segmentation']

        # set this segmentation a random color
        image[seg] = colors[i]
    return image


def process_image(img_path, out_path, mask_generator):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # mask generator wants the default uint8 image
    masks = mask_generator.generate(image)

    # 初始化边缘掩码图（与原图尺寸相同，初始为全黑）
    edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for mask_data in masks:
        # 获取掩码的二值矩阵（HxW，0为背景，1为前景）
        mask = mask_data["segmentation"].astype(np.uint8) * 255  # 转为0-255范围

        # 对掩码进行边缘检测（Canny算法）
        # 阈值可根据图像调整，以控制边缘检测的灵敏度
        edges = cv2.Canny(mask, threshold1=50, threshold2=150)

        # 将当前掩码的边缘合并到总边缘掩码图中（边缘像素设为255）
        edge_mask = cv2.bitwise_or(edge_mask, edges)

    # 保存或显示边缘掩码图
    cv2.imwrite(out_path.replace(".png","edge_mask.png"),edge_mask)

    # convert to float64
    image = image.astype(np.float64) / 255
    seg = draw_segmentation(masks)

    # add segmentation image on top of original image
    image += transparency * seg

    # convert back to uint8 for display/save
    image = (255 * image).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("my img", image)
    # cv2.waitKey(-1)
    cv2.imwrite(out_path, image)

if __name__ == "__main__":
    # make sure output dir exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load SAM model + create mask generator
    sam = sam_model_registry[sam_model](checkpoint=sam_check)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    # process input directory
    for img in tqdm(os.listdir(in_dir)):

        # change extension of output image to .png
        out_img = Path(img).stem + ".png"
        out_img = os.path.join(out_dir, out_img)

        # if we can read/decode this file as an image
        in_img = os.path.join(in_dir, img)
        if cv2.haveImageReader(in_img):
            process_image(in_img, out_img, mask_generator)


# mask_generator_2 = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,  # Requires open-cv to run post-processing
# )
# masks2 = mask_generator_2.generate(image)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks2)
# plt.axis('off')
# plt.show()