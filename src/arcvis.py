# visualize_arcface.py
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision import transforms
from typing import List, Tuple, Optional
from matplotlib.animation import PillowWriter
from model import get_model
import cv2

def reshape_transform(tensor, height=12, width=12):
    """
    reshape transformer 的输出以适应 GradCAM 的输入要求
    Args:
        tensor: 形状为 [batch_size, num_patches, hidden_dim] 的张量
        height: 特征图的高度
        width: 特征图的宽度
    Returns:
        重塑后的张量，形状为 [batch_size, hidden_dim, height, width]
    """
    return tensor.permute(0, 3, 1, 2)

class ArcFaceWrapper(nn.Module):
    """
    包装 ArcFace 模型以适应 GradCAM
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.weight = model.arcface_loss.weight
        
    def forward(self, x):
        # 1. 获取 backbone 特征
        features = self.model.backbone.forward_features(x)
        
        # 2. 调整维度顺序
        features = features.permute(0, 3, 1, 2)
        
        # 3. 通过特征处理模块
        features = self.model.feature_module(features)
        
        # 4. 计算分类分数
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(features_norm, weight_norm)

def visualize_gradcam(
    model: nn.Module,
    image_path: str,
    image_output_dir: str,
    target_classes: Optional[List[int]] = None,
    device: Optional[torch.device] = None
):
    """改进的 GradCAM 可视化实现"""
    os.makedirs(image_output_dir, exist_ok=True)
    
    # 1. 图像预处理
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (384, 384))
    rgb_img = np.float32(rgb_img) / 255
    
    input_tensor = preprocess_image(
        rgb_img,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    
    if device:
        input_tensor = input_tensor.to(device)
    
    # 2. 模型包装
    wrapped_model = ArcFaceWrapper(model)
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    # 3. 选择目标层
    # 使用 backbone 最后一个 stage 的最后一个 block
    target_layer = [model.backbone.layers[-1].blocks[-1]]
    
    # 4. 初始化 GradCAM
    cam = GradCAM(
        model=wrapped_model,
        target_layers=target_layer,
        use_cuda=device.type == "cuda" if device else False,
        reshape_transform=reshape_transform  # 使用自定义的 reshape 函数
    )
    
    # 5. 确定目标类别
    num_classes = model.arcface_loss.num_classes
    target_classes = target_classes or range(num_classes)
    
    # 6. 生成每个类别的 GradCAM
    for class_idx in target_classes:
        # 计算 GradCAM
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # 生成可视化
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # 设置全局字体参数
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 12
        
        # 创建和保存图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 原始图像
        ax1.imshow(rgb_img)
        ax1.set_title("Original Image", fontsize=14, pad=10)
        ax1.axis('off')
        
        # GradCAM
        ax2.imshow(cam_image)
        ax2.set_title(f'GradCAM - Class {class_idx}', fontsize=14, pad=10)
        ax2.axis('off')
        
        # 调整布局并添加总标题
        plt.suptitle(f'Attention Visualization - Class {class_idx}', 
                    fontsize=16, y=0.95)
        
        # 设置更大的间距
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图像，确保高质量输出
        save_path = os.path.join(image_output_dir, f'gradcam_class_{class_idx}.png')
        plt.savefig(save_path, 
                   dpi=150, 
                   bbox_inches='tight',
                   pad_inches=0.5,
                   facecolor='white')
        plt.close(fig)
        
        print(f"保存类别 {class_idx} 的 GradCAM 到 {save_path}")

def create_gif(
    image_output_dir: str, 
    gif_output_dir: str, 
    duration: int = 1000
) -> None:
    """
    为每张图像的 GradCAM 可视化结果创建 GIF。
    
    Args:
        image_output_dir: GradCAM 图像保存的目录
        gif_output_dir: GIF 保存的目录
        duration: 每帧显示时间(毫秒)
    """
    os.makedirs(gif_output_dir, exist_ok=True)
    
    # 获取所有图像目录
    image_dirs = [
        d for d in os.listdir(image_output_dir) 
        if os.path.isdir(os.path.join(image_output_dir, d))
    ]
    
    for img_dir in image_dirs:
        img_path = os.path.join(image_output_dir, img_dir)
        # 获取所有 gradcam 图像并按类别排序
        class_files = sorted(
            glob.glob(os.path.join(img_path, 'gradcam_class_*.png')),
            key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0])
        )
        
        if not class_files:
            print(f"在 {img_path} 中未找到 GradCAM 图像,跳过...")
            continue
            
        # 创建 GIF
        frames = []
        for file in class_files:
            img = Image.open(file)
            frames.append(img.copy())
            img.close()
            
        gif_path = os.path.join(gif_output_dir, f'{img_dir}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"保存 {img_dir} 的 GIF 到 {gif_path}")

def main():
    """
    主函数,集成 GradCAM 可视化和 GIF 创建功能。
    """
    # 参数设置
    max_img = 2  # 处理图像的最大数量
    image_dir = '/home/visllm/program/plant/Project/data/val/images'  # 输入图像目录
    checkpoint_path = '/home/visllm/program/plant/Project/checkpoints_swin_enhanced_arcface_final/best_model.pth'  # 模型检查点路径
    output_dir = '/home/visllm/program/plant/Project/outputs/arcmodel'  # GradCAM 图像保存目录
    gif_output_dir = '/home/visllm/program/plant/Project/gifs'   # GIF 保存目录
    num_classes = 6  # 类别数量
    duration = 1000  # GIF 每帧显示时间
    
    # 获取图像路径
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if len(image_paths) == 0:
        image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    image_paths = image_paths[:max_img]
    
    if len(image_paths) == 0:
        print(f"在 {image_dir} 中未找到图像,退出.")
        return
        
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gif_output_dir, exist_ok=True)
    
    # 加载训练好的模型
    model = get_model(
        num_classes=num_classes,
        pretrained=True,
        model_name='swinv2',
        use_arcface=True,
        feature_dim=512
    )
    
    # 加载模型权重
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cuda:1' if torch.cuda.is_available() else 'cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 对每张图像进行 GradCAM 可视化
    print("正在对图像进行 GradCAM 可视化...")
    for idx, image_path in enumerate(image_paths):
        print(f"处理图像 {idx+1}/{len(image_paths)}: {image_path}")
        image_output_dir = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(image_path))[0]
        )
        os.makedirs(image_output_dir, exist_ok=True)
        
        try:
            visualize_gradcam(
                model=model,
                image_path=image_path,
                image_output_dir=image_output_dir,
                device=device
            )
        except Exception as e:
            print(f"处理 {image_path} 失败: {e}")
    
    # 创建 GIF
    print("正在创建 GIF...")
    create_gif(output_dir, gif_output_dir, duration)
    
    print(f"所有 GradCAM 可视化结果已保存到 {output_dir}")
    print(f"GIF 已保存到 {gif_output_dir}")

if __name__ == '__main__':
    main()