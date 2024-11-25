# import os
#
# # 原始文件夹路径
# root_dir = r"D:\\NUS_proj\\Bonus\\datasets\\Georgia\\Unmasked_processed"
#
# # 遍历子文件夹并重命名
# for i in range(1, 51):
#     old_name = os.path.join(root_dir, f"s{i:02}")
#     new_name = os.path.join(root_dir, str(i))
#
#     if os.path.exists(old_name):
#         os.rename(old_name, new_name)
#         print(f"Renamed {old_name} to {new_name}")
#     else:
#         print(f"{old_name} does not exist")

#
# import os
# import cv2
#
# def convert_images_to_rgb(root_dir):
#     # 遍历所有子文件夹
#     for subdir, dirs, files in os.walk(root_dir):
#         for file in files:
#             img_path = os.path.join(subdir, file)
#             image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#
#             # 检查图像是否正确读取
#             if image is None:
#                 print(f"Warning: Image at path {img_path} could not be read")
#                 continue
#
#             # 如果图像是灰度图像，转换为三通道图像
#             if len(image.shape) == 2 or image.shape[2] == 1:
#                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#                 cv2.imwrite(img_path, image)  # 覆盖原图像
#                 print(f"Converted {img_path} to RGB")
#
# # 使用函数
# root_dir = r"D:\\NUS_proj\\Bonus\\datasets\\Georgia\\Unmasked_processed"
# convert_images_to_rgb(root_dir)


# import os
# import re
#
# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
#
# def rename_images_in_subfolders(root_dir):
#     # 遍历所有子文件夹
#     for subdir, dirs, files in os.walk(root_dir):
#         for dir in dirs:
#             dir_path = os.path.join(subdir, dir)
#             image_files = sorted(os.listdir(dir_path), key=natural_sort_key)  # 使用自然排序
#
#             # 重新命名文件
#             for idx, file in enumerate(image_files):
#                 old_path = os.path.join(dir_path, file)
#                 new_filename = f"{idx + 1}.jpg"
#                 new_path = os.path.join(dir_path, new_filename)
#
#                 os.rename(old_path, new_path)
#                 print(f"Renamed {old_path} to {new_path}")
#
# # 使用函数
# root_dir = r"D:\\NUS_proj\\Bonus\\datasets\\Georgia\\Unmasked_processed"
# rename_images_in_subfolders(root_dir)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_radar(data, title, categories, values, labels):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    for label, value in zip(labels, values):
        value += value[:1]
        ax.plot(angles, value, linewidth=2, linestyle='solid', label=label)
        ax.fill(angles, value, alpha=0.25)

    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

def main():
    # 读取数据
    df = pd.read_csv('acc.csv')

    datasets = df['Dataset'].unique()
    methods = df['Method'].unique()

    for dataset in datasets:
        data = df[df['Dataset'] == dataset]
        categories = methods
        values = [data[data['Method'] == method]['Accuracy'].values[0] for method in methods]
        plot_radar(data, dataset, categories, [values], methods)

if __name__ == '__main__':
    main()

