import torch
import torch.nn as nn
import timm
from efficientnet_pytorch import EfficientNet
import os
import torch.nn.functional as F
from enhanced_modules import STN, PSA_s
from lossfunction import MultiLabelArcFaceLoss


class MambaOutEnhanced(nn.Module):
    def __init__(
        self,
        num_classes=6,
        pretrained=True,
        model_dir='/home/visllm/.cache/huggingface/evaclip',
        dropout_rate=0.5
    ):
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

class SwinV2Enhanced(nn.Module):
    """
    SwinV2模型类
    
    特点：
    1. 使用本地预训练的SwinV2模型
    2. 保持原始模型结构不变
    3. 仅修改最后的分类头
    
    参数：
        num_classes (int): 输出类别数，默认为6
        pretrained (bool): 是否使用预训练权重
        model_dir (str): 本地模型权重文件路径
    """
    def __init__(self, num_classes=6, pretrained=True, model_dir='/home/visllm/.cache/huggingface/swinv2'):
        super(SwinV2Enhanced, self).__init__()
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"指定的本地模型路径不存在: {model_dir}")
        
        # 直接加载完整的模型，包括分类头
        self.model = timm.create_model(
            'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k', 
            pretrained=False,
            num_classes=num_classes  # 直接设置输出类别数
        )
        
        # 如果指定使用预训练权重，则加载本地权重
        checkpoint = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location='cpu')
        checkpoint_filtered = {
            k: v for k, v in checkpoint.items() 
            if not k.endswith('attn_mask') 
            and not k.startswith('head')  # 排除分类头权重
        }
        # 加载权重并打印加载信息
        missing, unexpected = self.model.load_state_dict(checkpoint_filtered, strict=False)
        print(f"已加载本地模型权重：{model_dir}")
        if len(missing) > 0:
            print(f"缺失的权重：{missing}")
        if len(unexpected) > 0:
            print(f"未预期的权重：{unexpected}")

class PSANFeatureModule(nn.Module):
    """整合PSA注意力机制的特征处理模块"""
    def __init__(self, in_channels, feature_dim=512):
        super(PSANFeatureModule, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.PReLU()
        
        # PSA attention module
        self.psa = PSA_s(in_channels, in_channels)
        
        # Second conv block
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.PReLU()
        
        # Feature reduction path
        self.feature_path = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(in_channels, 768),
            nn.BatchNorm1d(768),
            nn.PReLU(),
            nn.Linear(768, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.PReLU(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模块的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # PSA attention
        x = self.psa(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Feature reduction
        x = self.feature_path(x)
        
        return x

class PSANArcFaceModel(nn.Module):
    def __init__(
        self,
        base_model,
        num_classes,
        feature_dim=512,
        scale=30.0,
        margin=0.5
    ):
        super(PSANArcFaceModel, self).__init__()
        
        # 验证base_model
        if not hasattr(base_model, "model"):
            raise ValueError("base_model must be a SwinV2Enhanced instance with a 'model' attribute.")
        
        # 提取backbone
        self.backbone = base_model.model
        self.in_features = self.backbone.num_features
        
        # 移除原始分类头
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        # 替换为PSAN特征模块
        self.feature_module = PSANFeatureModule(1024, feature_dim)
        print("PSA已注入！")
        self.feature_dim = feature_dim
        
        # 多标签ArcFace Loss
        self.arcface_loss = MultiLabelArcFaceLoss(
            in_features=feature_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin
        )
    
    def forward(self, x, labels=None):
        """
        前向传播
        Args:
            x: 输入图像
            labels: 如果是训练阶段，提供标签(多标签矩阵)；推理阶段为None
        Returns:
            训练阶段: (loss, logits)
            推理阶段: features
        """
        # 1. 提取backbone特征
        features = self.backbone.forward_features(x)
        
        # 2. 调整维度顺序 [B, H, W, C] -> [B, C, H, W]
        features = features.permute(0, 3, 1, 2)
        
        # 3. 通过PSAN特征处理模块
        features = self.feature_module(features)
        
        # 4. 根据阶段返回不同结果
        if labels is None:
            return features
        else:
            loss, logits = self.arcface_loss(features, labels)
            return loss, logits

def get_model(num_classes=6, 
              pretrained=True, 
              model_name='swinv2', 
              use_arcface=False, 
              feature_dim=512):
    """
    获取模型实例
    
    参数:
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重
        model_name (str): 模型名称
        use_arcface (bool): 是否使用 ArcFace 架构
        feature_dim (int): ArcFace 的特征维度
    
    返回:
        nn.Module: 模型实例
    """
    # 创建基础模型
    if model_name == 'swinv2':
        base_model = SwinV2Enhanced(
            num_classes=num_classes if not use_arcface else feature_dim,
            pretrained=pretrained
        )
    elif model_name == 'mambaout':
        base_model = MambaOutEnhanced(
            num_classes=num_classes if not use_arcface else feature_dim,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    if use_arcface:
        model = PSANArcFaceModel(
            base_model=base_model,
            num_classes=num_classes,
            feature_dim=feature_dim
        )
    else:
        model = base_model
        
    return model
