import torch.nn as nn
import torch
import timm
from efficientnet_pytorch import EfficientNet
import os
from typing import Tuple

# 导入自定义模块
from enhanced_modules import STN


class MambaOutEnhanced(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, model_dir='/home/visllm/.cache/huggingface/evaclip', dropout_rate=0.5):
        super(MambaOutEnhanced, self).__init__()
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"指定的本地模型路径不存在: {model_dir}")
        
        # 加载基础模型
        self.base_model = timm.create_model('eva_giant_patch14_224.clip_ft_in1k', pretrained=False)
        checkpoint = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location='cpu')
        self.base_model.load_state_dict(checkpoint)
        print(f"加载本地模型权重：{model_dir}")
        
        # 移除原始分类头
        self.base_model.head = nn.Identity()
        
        # 获取基础模型的输出特征维度
        in_features = self.base_model.num_features  

        # 添加空间变换网络 (STN)
        self.stn = STN()

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播。

        Args:
            x (Tensor): 输入图像张量，形状为 (batch_size, 3, H, W)。

        Returns:
            Tensor: 分类输出，形状为 (batch_size, num_classes)。
        """
        # 通过 STN 增强空间变换
        x = self.stn(x)
        
        # 提取特征
        features = self.base_model(x)  # 形状: (batch_size, in_features)
        
        # 通过 Dropout 层
        features = self.dropout(features)
        
        # 分类
        out = self.classifier(features)  # 形状: (batch_size, num_classes)
        return out


def get_model(num_classes=6, pretrained=True, model_name='original'):
    """
    根据 model_name 返回指定模型的实例。
    当 model_name='swin' 时加载 Swin Transformer；
    当 model_name='mambaout' 时加载 MambaOutEnhanced 模型；
    否则使用原先的 EfficientNet-B0。
    """
    if model_name == 'swin':
        # 使用 Swin Transformer
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif model_name == 'mambaout':
        # 使用增强后的 MambaOut 模型
        model = MambaOutEnhanced(num_classes=num_classes, pretrained=pretrained)
    else:
        # 使用原始的 EfficientNet
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            model = EfficientNet.from_name('efficientnet-b0')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)

    return model
