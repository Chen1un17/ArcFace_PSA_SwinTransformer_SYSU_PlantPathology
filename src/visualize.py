import os
import glob
import torch

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import cv2
from torchvision.transforms import ToTensor, Resize, Compose
from matplotlib.animation import FuncAnimation
from collections import OrderedDict
from matplotlib.animation import PillowWriter
# 假设以下函数和类已在您的项目中实现
from visualisation.core import GradCam
from visualisation.core.utils import device, image_net_preprocessing, image_net_postprocessing
from utils import tensor2img

from efficientnet_pytorch import EfficientNet

#########################################
# 函数定义：从checkpoint加载EfficientNet模型
#########################################
def load_trained_efficientnet(model_name, checkpoint_path, num_classes=6):
    model = EfficientNet.from_name(model_name)
    in_features = model._fc.in_features
    import torch.nn as nn
    model._fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

###########################
# 参数设置
###########################

max_img = 10

path = '/home/visllm/program/plant/Project/data/val/images'

checkpoint_path = '/home/visllm/program/plant/Project/checkpoints/best_model.pth'

num_classes = 6

image_paths = glob.glob(os.path.join(path, '*.jpg'))
if len(image_paths) == 0:
    image_paths = glob.glob(os.path.join(path, '*.png'))

image_paths = image_paths[:max_img]

images = [PIL.Image.open(p).convert('RGB') for p in image_paths]

# 预处理图像
inputs = [Compose([Resize((224,224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in images]
inputs = [i.to(device) for i in inputs]

model_outs = OrderedDict()

# 将原始图像resize到224x224，用于后续对比展示
images_resized = [cv2.resize(np.array(x),(224,224)) for x in images]

# 加载训练好的模型
model = load_trained_efficientnet('efficientnet-b0', checkpoint_path, num_classes=num_classes)

# 实例化GradCam
vis = GradCam(model, device)

# 对每张图像进行GradCam可视化
model_name = 'Trained EfficientNet-B0'
model_outs[model_name] = []
print("正在对图像进行GradCam可视化...")
for idx, inp in enumerate(inputs):
    # 打印进度
    print(f"Processing image {idx+1}/{len(inputs)}: {image_paths[idx]}")
    target_layer = model._conv_head
    cam_result = vis(inp, target_layer, postprocessing=image_net_postprocessing)[0]
    cam_img = tensor2img(cam_result)
    model_outs[model_name].append(cam_img)

del model
torch.cuda.empty_cache()

# 创建画布
fig_width = 2 + len(model_outs) * 2
fig, axs = plt.subplots(1, 1 + len(model_outs), figsize=(fig_width, 4))


if not isinstance(axs, np.ndarray):
    axs = [axs]
if len(axs) == 1:
    # 只有一个子图时进行特殊处理
    axs = [axs[0], axs[0]]

ax_orig = axs[0]
ax_models = axs[1:]

# update 函数用于动画逐帧更新
def update(frame):
    # 清空子图
    for ax in axs:
        ax.clear()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axis('off')

    # 显示原始图像
    ax_orig.imshow(images_resized[frame])
    ax_orig.set_title("Original Image")

    # 显示模型GradCam结果
    for ax, (m_name, m_out_list) in zip(ax_models, model_outs.items()):
        ax.imshow(m_out_list[frame])
        ax.set_title(m_name)

    # 在控制台打印当前帧，以便实时跟进进度
    print(f"Animating frame {frame+1}/{len(images_resized)}")

    return axs

ani = FuncAnimation(fig, update, frames=range(len(images_resized)), interval=1000, blit=False)

save_path = '/home/visllm/program/plant/Project/outputs/example.gif'

ani.save(save_path, writer=PillowWriter(fps=1))

print(f"GradCam可视化结果已保存到 {save_path}")
