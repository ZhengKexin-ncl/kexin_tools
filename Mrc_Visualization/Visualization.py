import numpy as np
import mrcfile
import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import os

def load_image(file_path):
    """
    加载MRC格式的图像数据
    :param file_path: 图像文件路径
    :return: 图像数据数组
    """
    with mrcfile.open(file_path, permissive=True) as mrc:
        image_data = mrc.data
    return image_data

def stretch(image):
    """
    将图像数据拉伸到0-255范围
    :param image: 输入图像数组
    :return: 拉伸后的图像数组
    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return image.astype(np.uint8)

def load_annotations(file_path):
    """
    加载标注数据
    :param file_path: 标注文件路径
    :return: 标注数据的Pandas DataFrame
    """
    annotations = pd.read_csv(file_path, delimiter='\t', header=None, names=['class_id', 'x', 'y', 'z'])
    return annotations

def get_slice(image_data, axis, index):
    """
    获取图像在指定轴上的切片
    :param image_data: 图像数据数组
    :param axis: 切片轴（'xy', 'xz', 'yz'）
    :param index: 切片索引
    :return: 切片图像数组
    """
    if axis == 'xy':
        return image_data[index, :, :]
    elif axis == 'xz':
        return image_data[:, index, :]
    elif axis == 'yz':
        return image_data[:, :, index]

def generate_random_colors(num_colors):
    """
    生成指定数量的随机颜色
    :param num_colors: 颜色数量
    :return: 颜色列表
    """
    colors = []
    for i in range(num_colors):
        color = tuple(random.randint(0, 255) / 255.0 for _ in range(3))
        colors.append(color)
    return colors

def annotate_image(slice_image, annotations, axis, index, colors, diameter=10, circle_width=2):
    """
    在切片图像上绘制标注圆形
    :param slice_image: 切片图像数组
    :param annotations: 标注数据的Pandas DataFrame
    :param axis: 切片轴（'xy', 'xz', 'yz'）
    :param index: 切片索引
    :param colors: 不同类别的颜色
    :param diameter: 圆形的直径
    :param circle_width: 圆形的线宽
    :return: 带有标注的RGB图像数组
    """
    rgb_image = cv2.cvtColor(slice_image, cv2.COLOR_GRAY2RGB)
    radius = diameter / 2
    for _, row in annotations.iterrows():
        class_id, x, y, z = row['class_id'], row['x'], row['y'], row['z']
        color = tuple(int(c * 255) for c in colors[int(class_id) % len(colors)])  # 选择对应类别的颜色
        if axis == 'xy' and abs(z - index) <= radius:
            adjusted_diameter = diameter * math.sqrt(1 - (z - index)**2 / radius**2)
            cv2.circle(rgb_image, (int(y), int(x)), int(adjusted_diameter), color, circle_width)
        elif axis == 'xz' and abs(y - index) <= radius:
            adjusted_diameter = diameter * math.sqrt(1 - (y - index)**2 / radius**2)
            cv2.circle(rgb_image, (int(z), int(x)), int(adjusted_diameter), color, circle_width)
        elif axis == 'yz' and abs(x - index) <= radius:
            adjusted_diameter = diameter * math.sqrt(1 - (x - index)**2 / radius**2)
            cv2.circle(rgb_image, (int(z), int(y)), int(adjusted_diameter), color, circle_width)
    return rgb_image

def draw_grid(image, grid_size=50, color=(0, 255, 0)):
    """
    在图像上绘制网格
    :param image: 输入图像数组
    :param grid_size: 网格大小
    :param color: 网格颜色
    :return: 带有网格的图像数组
    """
    h, w = image.shape[:2]
    for i in range(0, w, grid_size):
        cv2.line(image, (i, 0), (i, h), color, 1)
    for i in range(0, h, grid_size):
        cv2.line(image, (0, i), (w, i), color, 1)
    return image

def add_scale_bar(image, scale_length=100, scale_text='100nm', position=(10, 10), color=(255, 255, 255)):
    """
    在图像上添加比例尺
    :param image: 输入图像数组
    :param scale_length: 比例尺长度（像素）
    :param scale_text: 比例尺文本
    :param position: 比例尺位置
    :param color: 比例尺颜色
    :return: 带有比例尺的图像数组
    """
    h, w = image.shape[:2]
    x, y = position
    cv2.line(image, (x, y), (x + scale_length, y), color, 2)
    cv2.putText(image, scale_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def add_legend(colors, labels, save_path):
    """
    创建图例并保存为图像文件
    :param colors: 颜色列表
    :param labels: 标签列表
    :param save_path: 图例保存路径
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    ax.legend(legend_patches, labels, loc='center')
    ax.axis('off')
    plt.savefig(save_path)
    plt.close(fig)

def save_image(image, save_path):
    """
    保存图像到指定路径
    :param image: 输入图像数组
    :param save_path: 保存路径
    """
    plt.imsave(save_path, image)

# 示例
image_path = r'D:\dataset\test_visualization\model_0.mrc'  # 图像文件路径
annotations_path = r'D:\dataset\test_visualization\model_0.coords'  # 标注文件路径
save_folder = r'D:\dataset\test_visualization\output'  # 输出文件夹路径

# 自定义 x, y, z 数值
custom_x = 100
custom_y = 150
custom_z = 200

# 加载图像数据并进行拉伸
image_data = load_image(image_path)
stretched_image = stretch(image_data)

# 加载标注数据
annotations = load_annotations(annotations_path)

# 统计分类数
num_classes = annotations['class_id'].nunique()

# 生成不同类别的颜色
colors = generate_random_colors(num_classes)

# 获取图像在不同轴上的切片
slice_xy = get_slice(stretched_image, 'xy', custom_z)
slice_xz = get_slice(stretched_image, 'xz', custom_y)
slice_yz = get_slice(stretched_image, 'yz', custom_x)

# 设置标注的直径和线宽
diameter = 10
circle_width = 2

# 在切片图像上绘制标注圆形
annotated_slice_xy = annotate_image(slice_xy, annotations, 'xy', custom_z, colors, diameter, circle_width)
annotated_slice_xz = annotate_image(slice_xz, annotations, 'xz', custom_y, colors, diameter, circle_width)
annotated_slice_yz = annotate_image(slice_yz, annotations, 'yz', custom_x, colors, diameter, circle_width)

# 在切片图像上绘制网格
annotated_slice_xy = draw_grid(annotated_slice_xy)
annotated_slice_xz = draw_grid(annotated_slice_xz)
annotated_slice_yz = draw_grid(annotated_slice_yz)

# 在切片图像上添加比例尺
annotated_slice_xy = add_scale_bar(annotated_slice_xy)
annotated_slice_xz = add_scale_bar(annotated_slice_xz)
annotated_slice_yz = add_scale_bar(annotated_slice_yz)

# 创建输出文件夹
os.makedirs(save_folder, exist_ok=True)

# 保存图像
save_image(annotated_slice_xy, os.path.join(save_folder, 'annotated_slice_xy.png'))
save_image(annotated_slice_xz, os.path.join(save_folder, 'annotated_slice_xz.png'))
save_image(annotated_slice_yz, os.path.join(save_folder, 'annotated_slice_yz.png'))

# 添加图例并保存
unique_classes = annotations['class_id'].unique()
labels = [f'Class {int(cls)}' for cls in unique_classes]
add_legend(colors, labels, os.path.join(save_folder, 'legend.png'))

# 显示标注后的图像
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

axs[0, 0].imshow(annotated_slice_xy)
axs[0, 0].set_title('XY Slice')

axs[0, 1].imshow(annotated_slice_xz)
axs[0, 1].set_title('XZ Slice')

axs[1, 0].imshow(annotated_slice_yz)
axs[1, 0].set_title('YZ Slice')

# 添加图例到右下角
legend_image = plt.imread(os.path.join(save_folder, 'legend.png'))
axs[1, 1].imshow(legend_image)
axs[1, 1].set_title('Legend')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
