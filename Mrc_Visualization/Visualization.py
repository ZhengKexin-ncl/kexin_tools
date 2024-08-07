import numpy as np
import mrcfile
import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import os


def load_image(file_path):
    with mrcfile.open(file_path, permissive=True) as mrc:
        image_data = mrc.data
    return image_data


def stretch(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return image.astype(np.uint8)


def load_annotations(file_path):
    annotations = pd.read_csv(file_path, delimiter='\t', header=None, names=['class_id', 'x', 'y', 'z'])
    return annotations


def get_slice(image_data, axis, index):
    if axis == 'xy':
        return image_data[index, :, :]
    elif axis == 'xz':
        return image_data[:, index, :]
    elif axis == 'yz':
        return image_data[:, :, index]


def generate_random_colors(num_colors):
    colors = []
    for i in range(num_colors):
        color = tuple(random.randint(0, 255) / 255.0 for _ in range(3))
        colors.append(color)
    return colors


def annotate_image(slice_image, annotations, axis, index, colors, diameter=10, circle_width=2):
    rgb_image = cv2.cvtColor(slice_image, cv2.COLOR_GRAY2RGB)
    radius = diameter / 2
    for _, row in annotations.iterrows():
        class_id, x, y, z = row['class_id'], row['x'], row['y'], row['z']
        color = tuple(int(c * 255) for c in colors[int(class_id) % len(colors)])  # 选择对应类别的颜色
        if axis == 'xy' and abs(z - index) <= radius:
            adjusted_diameter = diameter * math.sqrt(1 - (z - index) ** 2 / radius ** 2)
            cv2.circle(rgb_image, (int(y), int(x)), int(adjusted_diameter), color, circle_width)
        elif axis == 'xz' and abs(y - index) <= radius:
            adjusted_diameter = diameter * math.sqrt(1 - (y - index) ** 2 / radius ** 2)
            cv2.circle(rgb_image, (int(z), int(x)), int(adjusted_diameter), color, circle_width)
        elif axis == 'yz' and abs(x - index) <= radius:
            adjusted_diameter = diameter * math.sqrt(1 - (x - index) ** 2 / radius ** 2)
            cv2.circle(rgb_image, (int(z), int(y)), int(adjusted_diameter), color, circle_width)
    return rgb_image


def add_legend(colors, labels, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in
                      colors]
    ax.legend(legend_patches, labels, loc='center')
    ax.axis('off')
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_image(image, save_path):
    plt.imsave(save_path, image, cmap='gray')


def pad_image_with_white(image, pad_width):
    height, width = image.shape[:2]
    white_padding = np.ones((height, pad_width, 3), dtype=np.uint8) * 255
    padded_image = np.concatenate((image, white_padding), axis=1)
    return padded_image


def overlay_legend_on_image(image, legend, offset_x):
    height, width = image.shape[:2]
    legend_height, legend_width = legend.shape[:2]
    if legend_height > height:
        # 调整图例高度，使其与图像高度相等
        legend = resize_image(legend, height)
        legend_height, legend_width = legend.shape[:2]
    overlay_image = image.copy()
    overlay_image[:legend_height, offset_x:offset_x + legend_width] = legend
    return overlay_image


def resize_image(image, new_height):
    height, width = image.shape[:2]
    new_width = int(new_height * width / height)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


# 示例
image_path = r'D:\dataset\test_visualization\model_0.mrc'
annotations_path = r'D:\dataset\test_visualization\model_0.coords'
save_folder = r'D:\dataset\test_visualization\output'

custom_x = 100
custom_y = 150
custom_z = 100

image_data = load_image(image_path)
stretched_image = stretch(image_data)

annotations = load_annotations(annotations_path)

num_classes = annotations['class_id'].nunique()
colors = generate_random_colors(num_classes)

slice_xy = get_slice(stretched_image, 'xy', custom_z)
slice_xz = get_slice(stretched_image, 'xz', custom_y)
slice_yz = get_slice(stretched_image, 'yz', custom_x)

diameter = 9
circle_width = 2

annotated_slice_xy = annotate_image(slice_xy, annotations, 'xy', custom_z, colors, diameter, circle_width)
annotated_slice_xz = annotate_image(slice_xz, annotations, 'xz', custom_y, colors, diameter, circle_width)
annotated_slice_yz = annotate_image(slice_yz, annotations, 'yz', custom_x, colors, diameter, circle_width)

os.makedirs(save_folder, exist_ok=True)

save_image(slice_xy, os.path.join(save_folder, 'original_slice_xy.png'))
save_image(slice_xz, os.path.join(save_folder, 'original_slice_xz.png'))
save_image(slice_yz, os.path.join(save_folder, 'original_slice_yz.png'))

save_image(annotated_slice_xy, os.path.join(save_folder, 'annotated_slice_xy.png'))
save_image(annotated_slice_xz, os.path.join(save_folder, 'annotated_slice_xz.png'))
save_image(annotated_slice_yz, os.path.join(save_folder, 'annotated_slice_yz.png'))

unique_classes = annotations['class_id'].unique()
labels = [f'Class {int(cls)}' for cls in unique_classes]
legend_path = os.path.join(save_folder, 'legend.png')
add_legend(colors, labels, legend_path)

legend_image = cv2.imread(legend_path)

# 对每个标注后的图片分别填充空白并覆盖图例
concatenated_images = []
for annotated_slice in [annotated_slice_xy, annotated_slice_xz, annotated_slice_yz]:
    annotated_height = annotated_slice.shape[0]
    # 调整图例高度，使其与图像高度相等
    if legend_image.shape[0] != annotated_height:
        legend_resized = resize_image(legend_image, annotated_height)
    else:
        legend_resized = legend_image

    legend_width = legend_resized.shape[1]
    annotated_width = annotated_slice.shape[1]
    padded_image = pad_image_with_white(annotated_slice, legend_width)

    # 将图例覆盖在空白区域的左侧，紧贴标注好的图片
    concatenated_image = overlay_legend_on_image(padded_image, legend_resized, annotated_width)
    concatenated_images.append(concatenated_image)

concatenated_image_xy, concatenated_image_xz, concatenated_image_yz = concatenated_images

plt.imsave(os.path.join(save_folder, 'concatenated_image_xy.png'), concatenated_image_xy / 255.0)
plt.imsave(os.path.join(save_folder, 'concatenated_image_xz.png'), concatenated_image_xz / 255.0)
plt.imsave(os.path.join(save_folder, 'concatenated_image_yz.png'), concatenated_image_yz / 255.0)

fig, axs = plt.subplots(3, 1, figsize=(18, 36))

axs[0].imshow(concatenated_image_xy)
axs[0].set_title('XY Slice with Legend')

axs[1].imshow(concatenated_image_xz)
axs[1].set_title('XZ Slice with Legend')

axs[2].imshow(concatenated_image_yz)
axs[2].set_title('YZ Slice with Legend')

plt.tight_layout()
plt.show()
