import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalBCELoss(nn.Module):
    """实现Focal Loss来处理标签不平衡问题"""
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelArcFaceLoss(nn.Module):
    """
    多标签版本的 ArcFace Loss:
      - 与单标签 ArcFace 不同之处在于：labels 为 multi-hot (batch_size x num_classes)，
        可能在同一样本中对多个类别同时为 positive。
      - 对所有正类位置添加 margin，即 phi = cos(theta) * cos_m - sin(theta) * sin_m；
        对负类保持原 cos(theta)。
      - 使用 BCEWithLogitsLoss 作为多标签损失。
      
    可选: easy_margin, 当 cos(theta) < 0 时是否直接用 cos(theta) 取代 phi。
         也保留了阈值 self.th 与修正项 self.mm 处理，以符合原生 arcface 对 large angle 的限制。
    """
    def __init__(self, 
                 in_features, 
                 num_classes, 
                 scale=30.0, 
                 margin=0.5, 
                 easy_margin=False):
        super(MultiLabelArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = scale
        self.m = margin
        self.easy_margin = easy_margin

        # 权重矩阵，形状 [num_classes, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 预计算常量
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features, labels):
        """
        Args:
            features: [B, in_features], 未必归一化; 这里再行 L2 归一化
            labels:   [B, num_classes], multi-hot (0/1)，表示每个类别是否为正类
        Returns:
            loss:    (scalar) BCE 多标签损失
            logits:  [B, num_classes], 做 sigmoid 前的 raw logits
        """
        # (1) L2 归一化
        features_norm = F.normalize(features, p=2, dim=1)     # [B, in_features]
        weight_norm   = F.normalize(self.weight, p=2, dim=1)  # [num_classes, in_features]

        # (2) cos(theta) = features_norm • weight_norm
        #     形状: [B, num_classes]
        cosine = F.linear(features_norm, weight_norm)

        # (3) sin(theta) = sqrt(1 - cos^2)
        sine = torch.sqrt((1.0 - cosine**2).clamp(min=1e-9))

        # (4) phi = cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # (5) 如果启用 easy_margin:
        #     当 cos(theta) < 0 时, 直接用原 cos(theta), 否则用 phi
        #     (这在大角度时可能更稳定)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 当 cos(theta) <= self.th (== cos(pi - m)) 时，用 (cosine - self.mm) 替换 phi
            # 用于避免 cos(theta + m) 在 theta 较大时反而变正
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # (6) 用 labels 进行逐元素选择:
        #     labels=1 的位置 => phi,   labels=0 => cos(theta)
        #     这样就对所有正类加了 margin，负类维持原 cos(theta)
        #     注意 labels.bool() 只对 0/1 矩阵生效
        outputs = torch.where(labels.bool(), phi, cosine)

        # (7) 乘以 scale
        outputs = outputs * self.s

        # (8) 计算多标签 BCE Loss
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        return loss, outputs
