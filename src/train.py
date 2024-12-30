import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment, InterpolationMode
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from collections import defaultdict

from dataset import PlantPathologyDataset
from model import get_model
from utils import calculate_metrics, save_checkpoint, calculate_batch_accuracy, set_seed
from lossfunction import FocalBCELoss, MultiLabelArcFaceLoss

def get_transforms(resize_size, is_training=True):
    """
    获取数据增强策略。对于训练集，使用 RandAugment 进行自动数据增强。
    参数:
        resize_size: 图像调整的目标大小
        is_training: 是否是训练模式
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            RandAugment(num_ops=2, magnitude=9),  # 使用正确的参数名称
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def tta_inference(model, image, device):
    """
    测试时增强(TTA)实现。
    
    返回多个增强版本的平均特征向量，而不是直接的预测结果。
    """
    model.eval()
    features_list = []
    
    transforms_list = [
        lambda x: x,  # 原始图像
        lambda x: torch.flip(x, dims=[3]),  # 水平翻转
        lambda x: torch.flip(x, dims=[2]),  # 垂直翻转
        lambda x: torch.rot90(x, k=1, dims=[2, 3])  # 90度旋转
    ]
    
    with torch.no_grad():
        for transform in transforms_list:
            transformed_image = transform(image)
            # 直接获取特征向量
            features = model(transformed_image)
            features_list.append(features)
    
    # 返回平均特征向量
    return torch.stack(features_list).mean(dim=0)

def train_epoch(model, loader, criterion, optimizer, device, scaler, accumulation_steps=2):
    """
    训练一个epoch，使用混合精度训练和梯度累积来提高效率。
    """
    model.train()
    running_loss = 0.0
    
    # 初始化每个类别的准确率统计（假设6个类别）
    running_class_accuracies = np.zeros(6)
    num_batches = 0
    
    all_targets = []
    all_outputs = []
    
    optimizer.zero_grad()
    progress_bar = tqdm(loader, desc="Training")
    
    for idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 混合精度训练
        with autocast():
            # 注意：model 输出的是特征向量，而不是 logits
            features = model(images)
            # criterion: MultiLabelArcFaceLoss(features, labels) => (loss, logits)
            loss, logits = criterion(features, labels)
            # 为了梯度累积，对 loss 做平均
            loss = loss / accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积判断
        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # ---- 准确率计算：对 logits 做 sigmoid ----
        # 只有经过 sigmoid 才能阈值分割多标签
        probs = torch.sigmoid(logits)
        
        # 计算每个类别的准确率
        batch_class_accuracies, batch_overall_accuracy = calculate_batch_accuracy(
            probs, labels
        )
        running_class_accuracies += batch_class_accuracies
        num_batches += 1
        
        # 累积loss（要用加回 *images.size(0) 还原之前除的 accumulation_steps）
        running_loss += loss.item() * accumulation_steps * images.size(0)
        
        all_targets.append(labels.detach().cpu().numpy())
        all_outputs.append(probs.detach().cpu().numpy())
        
        # 更新进度条信息
        progress_info = {
            'loss': f'{loss.item():.4f}',
            'avg_acc': f'{batch_overall_accuracy:.4f}'
        }
        for i, acc in enumerate(batch_class_accuracies):
            progress_info[f'c{i}_acc'] = f'{acc:.2f}'
        progress_bar.set_postfix(progress_info)
    
    # 计算 epoch 级别指标
    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    # 计算f1、mAP等
    f1_score, _, map_score = calculate_metrics(all_outputs, all_targets)
    
    # 计算每个类别平均准确率
    class_accuracies = running_class_accuracies / num_batches
    overall_accuracy = np.mean(class_accuracies)
    
    return {
        'loss': epoch_loss,
        'accuracy': overall_accuracy,
        'class_accuracies': class_accuracies.tolist(),
        'f1': f1_score,
        'map': map_score
    }

def eval_epoch(model, loader, criterion, device, use_tta=False):
    """
    验证一个epoch，支持测试时增强(TTA)。
    """
    model.eval()
    running_loss = 0.0
    running_class_accuracies = np.zeros(6)
    num_batches = 0
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取特征向量
            if use_tta:
                features = tta_inference(model, images, device)
            else:
                features = model(images)
            
            # ArcFace Loss => (loss, logits)
            loss, logits = criterion(features, labels)
            
            # 多标签推理：要对 logits 做 sigmoid
            probs = torch.sigmoid(logits)
            
            # 计算准确率
            batch_class_accuracies, batch_overall_accuracy = calculate_batch_accuracy(
                probs, labels
            )
            
            running_class_accuracies += batch_class_accuracies
            num_batches += 1
            
            running_loss += loss.item() * images.size(0)
            
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())  # 保存已经 sigmoid 后的概率
            
            progress_info = {
                'loss': f'{loss.item():.4f}',
                'avg_acc': f'{batch_overall_accuracy:.4f}'
            }
            for i, acc in enumerate(batch_class_accuracies):
                progress_info[f'c{i}_acc'] = f'{acc:.2f}'
            progress_bar.set_postfix(progress_info)
    
    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    # 计算评估指标
    f1_score, _, map_score = calculate_metrics(all_outputs, all_targets)
    
    class_accuracies = running_class_accuracies / num_batches
    overall_accuracy = np.mean(class_accuracies)
    
    return {
        'loss': epoch_loss,
        'accuracy': overall_accuracy,
        'class_accuracies': class_accuracies.tolist(),
        'f1': f1_score,
        'map': map_score
    }

def plot_training_history(history, save_dir):
    """
    绘制训练历史曲线，包括整体和每个类别的指标。
    
    该函数创建两组图表：
    1. 整体指标图表（损失、平均准确率、F1分数和mAP）
    2. 每个类别的准确率图表（可选）
    """
    # 基础指标绘图
    basic_metrics = ['loss', 'f1', 'map']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制基础指标
    for idx, metric in enumerate(basic_metrics):
        ax = axes[idx//2, idx%2]
        ax.plot(history[f'train_{metric}'], label='Train')
        ax.plot(history[f'val_{metric}'], label='Validation')
        ax.set_title(f'{metric.upper()} Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)
    
    # 绘制平均准确率
    ax = axes[1, 1]
    if isinstance(history['train_accuracy'][0], list):
        # accuracy是类别准确率的列表，计算平均值
        train_avg_acc = [np.mean(accs) for accs in history['train_accuracy']]
        val_avg_acc = [np.mean(accs) for accs in history['val_accuracy']]
        ax.plot(train_avg_acc, label='Train')
        ax.plot(val_avg_acc, label='Validation')
    else:
        # 如果是单个准确率值
        ax.plot(history['train_accuracy'], label='Train')
        ax.plot(history['val_accuracy'], label='Validation')
    ax.set_title('Average Accuracy Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # 绘制每个类别的准确率曲线
    if isinstance(history['train_accuracy'][0], list):
        class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 
                      'rust', 'complex', 'powdery_mildew']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, class_name in enumerate(class_names):
            ax = axes[idx]
            train_class_acc = [acc[idx] for acc in history['train_accuracy']]
            val_class_acc = [acc[idx] for acc in history['val_accuracy']]
            ax.plot(train_class_acc, label='Train')
            ax.plot(val_class_acc, label='Validation')
            ax.set_title(f'{class_name} Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_accuracy_curves.png'), dpi=300)
        plt.close()

def main():
    """主训练函数"""
    # 设置随机种子
    set_seed(42)
    
    # 配置参数与路径
    data_dir = '/home/visllm/program/plant/Project/data'
    train_csv = os.path.join(data_dir, 'processed_train_labels.csv')
    val_csv = os.path.join(data_dir, 'processed_val_labels.csv')
    train_images = os.path.join(data_dir, 'train', 'images')
    val_images = os.path.join(data_dir, 'val', 'images')
    
    # 训练参数
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-4
    num_classes = 6
    accumulation_steps = 2
    resize_size = 384
    
    # 创建实验目录
    checkpoint_dir = os.path.join(data_dir, f'../checkpoints_swin_enhanced_arcface')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 准备数据加载器
    train_dataset = PlantPathologyDataset(
        csv_file=train_csv,
        images_dir=train_images,
        transform=get_transforms(resize_size, is_training=True)
    )
    
    val_dataset = PlantPathologyDataset(
        csv_file=val_csv,
        images_dir=val_images,
        transform=get_transforms(resize_size, is_training=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型和训练组件
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = get_model(
        num_classes=num_classes,
        pretrained=True,
        model_name='swinv2',
        use_arcface=True,  # 启用 ArcFace
        feature_dim=512    # 设置特征维度
    )
    model = model.to(device)
    
    criterion = MultiLabelArcFaceLoss(
        in_features=512,  # 特征维度
        num_classes=num_classes,
        scale=30.0,
        margin=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    
    scaler = GradScaler()
    
    # 训练状态变量
    best_val_f1 = 0.0
    best_epoch = 0
    patience = 5
    counter = 0
    history = defaultdict(list)
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # 训练和验证
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                  device, scaler, accumulation_steps)
        val_metrics = eval_epoch(model, val_loader, criterion, device, use_tta=True)
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        for phase in ['train', 'val']:
            metrics = train_metrics if phase == 'train' else val_metrics
            for metric_name, value in metrics.items():
                history[f'{phase}_{metric_name}'].append(value)
        
        # 打印当前结果
        print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, mAP: {train_metrics['map']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, mAP: {val_metrics['map']:.4f}")
        
        # 模型保存和早停
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            counter = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'best_val_f1': best_val_f1,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model (epoch {epoch})")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
        
        # 定期保存检查点和最后几个epoch的模型
        if epoch % 10 == 0 or epoch > num_epochs - 5:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth'))
    
    # 训练结束后的工作
    # 1. 保存训练历史图表
    plot_training_history(history, checkpoint_dir)
    
    # 2. 保存训练配置和结果
    config_file = os.path.join(checkpoint_dir, 'training_config.txt')
    with open(config_file, 'w') as f:
        f.write("Training Configuration and Results\n")
        f.write("================================\n")
        f.write(f"Model: Swin Transformer V2\n")
        f.write(f"Image Size: {resize_size}x{resize_size}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Number of Epochs: {epoch}\n")
        f.write(f"Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch})\n")
        f.write(f"\nFinal Validation Metrics:\n")
        for metric_name, value in val_metrics.items():
            if isinstance(value, list):
                # 如果是类别准确率列表，分别保存每个类别的准确率
                class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 
                            'rust', 'complex', 'powdery_mildew']
                f.write(f"  {metric_name}:\n")
                for idx, class_acc in enumerate(value):
                    f.write(f"    {class_names[idx]}: {class_acc:.4f}\n")
            else:
                # 其他标量指标直接保存
                f.write(f"  {metric_name}: {value:.4f}\n")
    
    # 3. 保存训练历史数据
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(checkpoint_dir, 'training_history.csv'), index=False)
    
    print("\nTraining completed!")
    print(f"Best validation acc: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Model checkpoints and training history saved to: {checkpoint_dir}")

if __name__ == '__main__':
    main()