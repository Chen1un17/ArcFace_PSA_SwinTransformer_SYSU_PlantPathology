import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  

from dataset import PlantPathologyDataset
from model import get_model
from utils import calculate_metrics, save_checkpoint

from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

def train_epoch(model, loader, criterion, optimizer, device, scaler, max_grad_norm=1.0):
    """
    单个训练周期的逻辑：
    - 将模型置于训练模式
    - 对每个批次的数据进行前向传播、计算损失并反向传播更新参数
    - 使用混合精度加速计算
    - 引入梯度裁剪以防止梯度爆炸
    - 累积损失和预测结果，用于后续计算F1、Accuracy、mAP等指标
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # 引入梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        all_targets.append(labels.detach().cpu().numpy())
        all_outputs.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    f1, accuracy, map_score = calculate_metrics(all_outputs, all_targets)  
    return epoch_loss, f1, accuracy, map_score

def eval_epoch(model, loader, criterion, device):
    """
    单个验证/评估周期的逻辑：
    - 将模型置于评估模式
    - 不进行反向传播，仅计算输出和损失
    - 统计整体损失和预测结果，用于计算验证集上的F1、Accuracy、mAP指标
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_targets.append(labels.detach().cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    f1, accuracy, map_score = calculate_metrics(all_outputs, all_targets) 
    return epoch_loss, f1, accuracy, map_score

def main():
    """
    主函数流程：
    1. 配置参数和路径
    2. 创建数据集和数据加载器
    3. 初始化模型（可通过model_name选择使用Swin或EfficientNet）
    4. 使用BCEWithLogitsLoss作为损失函数
    5. 训练和验证模型，使用CosineAnnealingLR调整学习率并使用早停机制防止过拟合
    6. 记录训练和验证过程中损失与mAP，并最终保存曲线图
    """

    # 配置参数与路径
    data_dir = '/home/visllm/program/plant/Project/data'  
    train_csv = os.path.join(data_dir, 'processed_train_labels.csv')  
    val_csv = os.path.join(data_dir, 'processed_val_labels.csv')
    test_csv = os.path.join(data_dir, 'processed_test_labels.csv')

    train_images = os.path.join(data_dir, 'train', 'images')
    val_images = os.path.join(data_dir, 'val', 'images')
    test_images = os.path.join(data_dir, 'test', 'images') 

    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    num_classes = 6
    checkpoint_dir = os.path.join(data_dir, '../checkpoints_enhanced')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 在这里选择使用哪个模型架构
    # 'original' 表示使用EfficientNet，'swin'表示使用Swin Transformer，'mambaout'表示使用本地EVA模型
    model_name = 'mambaout'

    # 调整数据增强策略，避免过度变形
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),    
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 调整仿射变换参数
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和数据加载器
    train_dataset = PlantPathologyDataset(csv_file=train_csv, images_dir=train_images, transform=train_transform)
    val_dataset = PlantPathologyDataset(csv_file=val_csv, images_dir=val_images, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=num_classes, pretrained=False, model_name=model_name)  # pretrained=False 因为从本地加载
    model = model.to(device)

    # 使用BCEWithLogitsLoss作为损失函数
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 使用CosineAnnealingLR作为学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    scaler = GradScaler()

    best_val_f1 = 0.0
    best_epoch = 0
    patience = 10
    counter = 0

    train_losses = []
    val_losses = []
    train_maps = []
    val_maps = []

    # 开始训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_f1, train_acc, train_map = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_f1, val_acc, val_map = eval_epoch(model, val_loader, criterion, device)

        # 调度器步进
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maps.append(train_map)
        val_maps.append(val_map)

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train mAP: {train_map:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val mAP: {val_map:.4f} | Val Acc: {val_acc:.4f}")

        # 早停和保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, filename=checkpoint_path)
            print(f"保存最佳模型于 epoch {epoch}")
        else:
            counter += 1
            if counter >= patience:
                print("早停")
                break

    print(f"训练完成。最佳验证 F1 分数: {best_val_f1:.4f} 在 epoch {best_epoch}")

    # 绘制并保存损失曲线
    epochs_range = range(1, best_epoch + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_losses[:best_epoch], 'b-', label='Train Loss')
    plt.plot(epochs_range, val_losses[:best_epoch], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_curve_path = os.path.join(checkpoint_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"损失曲线已保存到 {loss_curve_path}")

    # 绘制并保存mAP曲线
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_maps[:best_epoch], 'b-', label='Train mAP')
    plt.plot(epochs_range, val_maps[:best_epoch], 'r-', label='Val mAP')
    plt.title('Training and Validation mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    map_curve_path = os.path.join(checkpoint_dir, 'map_curve.png')
    plt.savefig(map_curve_path)
    plt.close()
    print(f"mAP 曲线已保存到 {map_curve_path}")

if __name__ == '__main__':
    main()
