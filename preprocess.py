"""
    数据集预处理
"""

import cv2
import numpy as np
import dlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

# 数据集路径
unmasked_dir = '1'
processed_dir = '2'

# 初始化面部检测器和预测器
# 加载dlib的面部检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'D:\\NUS_proj\\Bonus\\1\\shape_predictor_68_face_landmarks.dat')


# 计算眼睛连线的旋转角度，用于校正面部的旋转
def calculate_rotation_angle(shape):
    # 获取左右眼的位置
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)

    # 计算眼睛连线的角度
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    return angle


# 根据给定的角度旋转图像
def rotate_image(image, angle):
    # 获取图像中心
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行仿射变换
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# 裁剪检测到的面部区域，并添加边距
def crop_face(image, d, margin=15):
    # 裁剪脸部区域并添加边距
    x, y, w, h = d.left(), d.top(), d.width(), d.height()
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = w + 2 * margin
    h = h + 2 * margin
    cropped_face = image[y:y + h, x:x + w]
    return cropped_face


def process_image(image_path, output_path, margin=15):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return None

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测面部
    dets = detector(gray, 1)
    if len(dets) == 0:
        print(f"No face detected in {image_path}")
        return None

    for k, d in enumerate(dets):
        shape = predictor(gray, d)

        # 计算旋转角度
        angle = calculate_rotation_angle(shape)

        # 旋转图像
        rotated_image = rotate_image(image, angle)

        # 检测旋转后的面部
        rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        dets_rotated = detector(rotated_gray, 1)
        if len(dets_rotated) == 0:
            print(f"No face detected in rotated image for {image_path}")
            return None

        # 裁剪脸部区域并添加边距
        cropped_face = crop_face(rotated_image, dets_rotated[0], margin)

        # 保存裁剪后的图像
        cv2.imwrite(output_path, cropped_face)

        return cropped_face


# 灰度图像均衡化
def enhance_contrast(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return None

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 直方图均衡化
    equalized = cv2.equalizeHist(gray)

    # 将均衡化后的灰度图像转换回BGR格式
    equalized_color = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    # 保存结果图像
    cv2.imwrite(output_path, equalized_color)

    return equalized_color


# 彩色图像均衡化
def equalize_yuv(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # 对Y通道进行直方图均衡化
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

    # 将图像从YUV转换回BGR颜色空间
    equalized_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return equalized_image


def equalize_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return None

    # 对图像进行直方图均衡化
    equalized = equalize_yuv(image)

    # 保存结果图像
    cv2.imwrite(output_path, equalized)

    return equalized


# 对特定区域高斯模糊
def apply_blur_to_mask_region(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return None

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测面部
    dets = detector(gray, 1)
    if len(dets) == 0:
        print(f"No face detected in {image_path}")
        return None

    for k, d in enumerate(dets):
        shape = predictor(gray, d)
        points = []
        for i in range(1, 16):
            point = [shape.part(i).x, shape.part(i).y]
            points.append(point)

        mask_a = [((shape.part(42).x), (shape.part(15).y)),
                  ((shape.part(27).x), (shape.part(27).y)),
                  ((shape.part(39).x), (shape.part(1).y))]
        fmask_a = points + mask_a
        fmask_a = np.array(fmask_a, dtype=np.int32)

        # 创建掩码区域
        mask_region = cv2.fillPoly(np.zeros_like(image), [fmask_a], (255, 255, 255))

        # 应用高斯模糊
        blurred = cv2.GaussianBlur(image, (21, 21), 0)

        # 将模糊区域应用到原图像
        mask = mask_region.astype(bool)
        image[mask] = blurred[mask]

    # 保存结果图像
    cv2.imwrite(output_path, image)

    return image


# 集成函数，核心
def process_images_in_directory(input_dir, output_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个人的图像列表
    for subdir, _, files in os.walk(input_dir):
        # 遍历此人的每张图像
        for file in tqdm(files, desc="Processing images"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                # 创建输出文件夹，如果不存在则创建
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 处理图像：裁剪人脸并保存结果
                cropped_face = process_image(input_path, output_path)
                if cropped_face is not None:
                    # 对图像进行直方图均衡化
                    equalized_image = equalize_image(output_path, output_path)
                    if equalized_image is not None:
                        # 增强图像对比度
                        # contrast_image = enhance_contrast(output_path, output_path)
                        # if contrast_image is not None:
                        # 对特定区域应用模糊
                        apply_blur_to_mask_region(output_path, output_path)

def preprocess_and_save_keypoints(dataset, predictor_path, save_path, margin=10, image_size=(224, 224)):
    """
    预处理数据集中的所有图像，提取嘴巴区域的关键点，并将其保存为 Pickle 文件。

    参数:
        dataset (Dataset): PyTorch 数据集对象。
        predictor_path (str): dlib 形状预测器的路径。
        save_path (str): 要保存的 Pickle 文件路径。
        margin (int, optional): 在嘴巴区域周围添加的边距。默认值为 10。
        image_size (tuple, optional): 输入图像的尺寸。默认值为 (224, 224)。
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    keypoints_dict = {}

    print("开始预处理关键点并保存为 Pickle 文件...")
    for idx in tqdm(range(len(dataset))):
        img_path = dataset.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                keypoints_dict[idx] = None
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 1)
            if len(dets) == 0:
                keypoints_dict[idx] = None
                continue
            # 假设每张图像只有一个人脸，取第一个检测到的人脸
            shape = predictor(gray, dets[0])
            mouth_points = [(shape.part(j).x, shape.part(j).y) for j in range(48, 68)]
            keypoints_dict[idx] = mouth_points
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            keypoints_dict[idx] = None

    # 保存关键点信息到 Pickle 文件
    with open(save_path, 'wb') as f:
        pickle.dump(keypoints_dict, f)

    print(f"关键点信息已保存到 {save_path}")


if __name__ == '__main__':
    # 处理目录下的所有图片
    process_images_in_directory(unmasked_dir, processed_dir)