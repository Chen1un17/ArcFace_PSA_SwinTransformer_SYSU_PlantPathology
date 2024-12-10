# visualize_and_create_gif.py

import os
import glob
import torch
import torch.nn.functional as F
import timm
from timm.models import create_model
from timm.utils import AttentionExtract
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import OrderedDict
from matplotlib.animation import PillowWriter

from model import get_model  # 确保 model.py 在同一目录下或在 PYTHONPATH 中

#########################################
# 函数定义：应用掩码到图像上
#########################################
def apply_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.5) -> np.ndarray:
    """
    应用掩码到图像上。
    """
    # 确保掩码和图像具有相同的形状
    mask = mask[:, :, np.newaxis]
    mask = np.repeat(mask, 3, axis=2)
    
    # 转换颜色为 numpy 数组
    color = np.array(color)
    
    # 应用掩码
    masked_image = image * (1 - alpha * mask) + alpha * mask * color[np.newaxis, np.newaxis, :] * 255
    return masked_image.astype(np.uint8)

#########################################
# 函数定义：根据 Rollout 方法计算最终的注意力图
#########################################
def rollout(attentions, discard_ratio, head_fusion, num_prefix_tokens=1):
    """
    根据 Rollout 方法计算最终的注意力图。
    """
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion.startswith('mean'):
                attention_heads_fused = attention.mean(dim=0)
            elif head_fusion == "max":
                attention_heads_fused = attention.amax(dim=0)
            elif head_fusion == "min":
                attention_heads_fused = attention.amin(dim=0)
            else:
                raise ValueError("Attention head fusion type Not supported")

            # 丢弃最低的注意力，但不丢弃前缀 token
            flat = attention_heads_fused.view(-1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices >= num_prefix_tokens]
            flat[indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    mask = result[0, num_prefix_tokens:]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    mask = mask / np.max(mask)
    return mask

#########################################
# 函数定义：对单个图像进行注意力可视化，并保存结果
#########################################
def visualize_attention(
        model: torch.nn.Module,
        extractor: AttentionExtract,
        image_path: str,
        image_output_dir: str,
        head_fusion: str = 'mean',
        discard_ratio: float = 0.9
    ):
    """
    对单个图像进行注意力可视化，并保存结果。
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # 获取模型的预处理配置
    config = model.pretrained_cfg

    # 解析 input_size
    input_size = config.get('input_size', (3, 224, 224))
    if isinstance(input_size, tuple):
        if len(input_size) == 3:
            _, height, width = input_size
            resize_size = (height, width)
        elif len(input_size) == 2:
            resize_size = input_size
        else:
            raise ValueError(f"Unexpected input_size format: {input_size}")
    elif isinstance(input_size, int):
        resize_size = input_size
    else:
        raise ValueError(f"Unexpected input_size format: {input_size}")

    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])

    # 预处理图像
    tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # 提取注意力图
    attention_maps = extractor(tensor)

    # 获取前缀 token 数量（通常为1个 class token）
    num_prefix_tokens = getattr(model, 'num_prefix_tokens', 1)  # 默认1

    # 创建子目录以保存当前图像的注意力图
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(image_output_dir, exist_ok=True)

    # 保存所有层的可视化
    attentions_for_rollout = []

    for layer_idx, (layer_name, attn_map) in enumerate(attention_maps.items(), start=1):
        print(f"Processing attention map for {layer_name} with shape {attn_map.shape}")
        attn_map = attn_map[0]  # 移除 batch 维度
        attentions_for_rollout.append(attn_map)

        # 根据 head_fusion 方法融合多个头的注意力
        if head_fusion == 'mean_std':
            attn_map = attn_map.mean(0) / attn_map.std(0)
        elif head_fusion == 'mean':
            attn_map = attn_map.mean(0)
        elif head_fusion == 'max':
            attn_map = attn_map.amax(0)
        elif head_fusion == 'min':
            attn_map = attn_map.amin(0)
        else:
            raise ValueError(f"Invalid head fusion method: {head_fusion}")

        # 使用第一个 token 的注意力（通常是 class token）
        attn_map = attn_map[0]

        # 移除 class token 的注意力
        attn_map = attn_map[1:]  # 只保留 patch tokens

        # 重新调整注意力图的形状
        num_patches = int(attn_map.shape[0] ** 0.5)
        if num_patches ** 2 != attn_map.shape[0]:
            raise ValueError(f"Number of patches {attn_map.shape[0]} is not a perfect square.")
        attn_map = attn_map.reshape(num_patches, num_patches)

        # 插值调整到图像大小
        attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).to(tensor.device)
        attn_map = F.interpolate(attn_map, size=(image_np.shape[0], image_np.shape[1]), mode='bilinear', align_corners=False)
        attn_map = attn_map.squeeze().cpu().numpy()

        # 归一化注意力图
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        # 创建可视化图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # 原始图像
        ax1.imshow(image_np)
        ax1.set_title("Original Image")
        ax1.axis('off')

        # 注意力图叠加
        masked_image = apply_mask(image_np, attn_map, color=(1, 0, 0), alpha=0.5)  # 红色掩码
        ax2.imshow(masked_image)
        ax2.set_title(f'Attention Map for {layer_name}')
        ax2.axis('off')

        plt.tight_layout()

        # 将图形保存为图像文件
        save_path = os.path.join(image_output_dir, f'attention_map_layer_{layer_idx}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved attention map for {layer_name} at {save_path}")

    # 计算 Rollout 注意力图
    rollout_mask = rollout(attentions_for_rollout, discard_ratio, head_fusion, num_prefix_tokens)

    # 创建 Rollout 可视化图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 原始图像
    ax1.imshow(image_np)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Rollout 注意力图叠加
    rollout_mask_pil = Image.fromarray((rollout_mask * 255).astype(np.uint8))
    rollout_mask_resized = np.array(rollout_mask_pil.resize((image_np.shape[1], image_np.shape[0]), Image.BICUBIC)) / 255.0
    masked_image = apply_mask(image_np, rollout_mask_resized, color=(1, 0, 0), alpha=0.5)  # 红色掩码
    ax2.imshow(masked_image)
    ax2.set_title('Attention Rollout')
    ax2.axis('off')

    plt.tight_layout()

    # 将图形保存为图像文件
    rollout_save_path = os.path.join(image_output_dir, 'attention_rollout.png')
    plt.savefig(rollout_save_path)
    plt.close(fig)
    print(f"Saved attention rollout at {rollout_save_path}")

#########################################
# 函数定义：创建 GIF
#########################################
def create_gif(image_output_dir, gif_output_dir, duration=1000):
    """
    为每张图像的注意力图生成 GIF。

    Args:
        image_output_dir (str): visualize_attention 保存注意力图像的主目录。
        gif_output_dir (str): 保存生成的 GIF 的目录。
        duration (int): 每帧显示的时间，单位为毫秒。
    """
    os.makedirs(gif_output_dir, exist_ok=True)

    # 遍历每个子目录（每张图像）
    image_dirs = [d for d in os.listdir(image_output_dir) if os.path.isdir(os.path.join(image_output_dir, d))]

    for img_dir in image_dirs:
        img_path = os.path.join(image_output_dir, img_dir)
        # 获取所有 attention_map_layer_X.png 文件，并按层数排序
        layer_files = sorted(
            glob.glob(os.path.join(img_path, 'attention_map_layer_*.png')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1].split('.')[0])
        )

        # 获取 rollout 图像
        rollout_file = os.path.join(img_path, 'attention_rollout.png')
        if os.path.exists(rollout_file):
            layer_files.append(rollout_file)

        # 检查是否有图像可用
        if not layer_files:
            print(f"No attention maps found in {img_path}. Skipping...")
            continue

        # 打开所有图像
        frames = []
        for file in layer_files:
            img = Image.open(file)
            frames.append(img.copy())
            img.close()

        # 创建 GIF
        gif_path = os.path.join(gif_output_dir, f'{img_dir}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved GIF for {img_dir} at {gif_path}")

#########################################
# 主函数
#########################################
def main():
    """
    主函数，集成可视化和 GIF 创建功能。
    """
    # 参数设置
    max_img = 2
    image_dir = '/home/visllm/program/plant/Project/data/val/images'  # 输入图像目录
    checkpoint_path = '/home/visllm/program/plant/Project/checkpoints/best_model.pth'  # 模型检查点路径
    num_classes = 6
    output_dir = '/home/visllm/program/plant/Project/outputs'  # 注意力图像保存目录
    gif_output_dir = '/home/visllm/program/plant/Project/gifs'  # GIF 保存目录
    head_fusion = 'mean'     # 可选: 'mean_std', 'mean', 'max', 'min'
    discard_ratio = 0.9
    duration = 1000  # GIF 每帧显示时间，单位毫秒

    # 获取图像路径
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if len(image_paths) == 0:
        image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    image_paths = image_paths[:max_img]

    if len(image_paths) == 0:
        print(f"No images found in {image_dir}. Exiting.")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gif_output_dir, exist_ok=True)

    # 加载训练好的模型
    model = get_model(num_classes=num_classes, pretrained=False, model_name='mambaout')  # 修改为您的模型名称

    # 加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location='cuda:2' if torch.cuda.is_available() else 'cpu')
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 准备注意力提取器
    extractor = AttentionExtract(model, method='fx')

    # 对每张图像进行注意力可视化
    print("正在对图像进行注意力可视化...")
    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
        image_output_dir_specific = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(image_output_dir_specific, exist_ok=True)
        try:
            visualize_attention(
                model=model,
                extractor=extractor,
                image_path=image_path,
                image_output_dir=image_output_dir_specific,
                head_fusion=head_fusion,
                discard_ratio=discard_ratio
            )
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

    # 创建 GIF
    print("正在创建 GIF...")
    create_gif(output_dir, gif_output_dir, duration)

    print(f"所有注意力可视化结果已保存到 {output_dir}，GIF 已保存到 {gif_output_dir}")

if __name__ == '__main__':
    main()
