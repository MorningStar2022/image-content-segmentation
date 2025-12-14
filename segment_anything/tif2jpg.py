import cv2
import numpy as np


def tif2img_opencv(tif_path, output_path):
    """
    单文件TIF转JPG/PNG（OpenCV版）
    :param tif_path: 输入TIF路径
    :param output_path: 输出路径（后缀为.jpg/.png）
    """
    try:
        # 读取TIF文件（cv2.IMREAD_UNCHANGED保留原深度/通道）
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("无法读取TIF文件，可能是格式不支持")

        # 处理16位深度TIF（转为8位）
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)  # 16位转8位

        # 保存为目标格式
        cv2.imwrite(output_path, img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95] if output_path.endswith(".jpg") else [cv2.IMWRITE_PNG_COMPRESSION,
                                                                                         6])
        print(f"转换成功：{output_path}")
    except Exception as e:
        print(f"转换失败：{tif_path}，错误：{str(e)}")


# 调用示例
if __name__ == "__main__":
    tif2img_opencv("test/4_300_300c_2480_3512.tif", "test/4_300_300c_2480_3512.jpg")
    # tif2img_opencv("input.tif", "output_opencv.png")