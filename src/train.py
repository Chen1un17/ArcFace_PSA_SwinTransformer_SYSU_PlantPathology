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

def train_epoch(model, loader, criterion, optimizer, device, scaler):
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
    f1, accuracy, map_score = calculate_metrics(all_outputs, all_targets)  # Modified to include mAP
    return epoch_loss, f1, accuracy, map_score  # Modified to return mAP

def main():
    # 配置参数
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
    checkpoint_dir = os.path.join(data_dir, '../checkpoints')  # 更新为检查点存储路径
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda:1")
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 初始化混合精度 scaler
    scaler = GradScaler()

    best_val_f1 = 0.0
    best_epoch = 0
    patience = 10
    counter = 0

    # 初始化记录变量
    train_losses = []
    val_losses = []
    train_maps = []
    val_maps = []

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_f1, train_acc, train_map = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_f1, val_acc, val_map = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maps.append(train_map)
        val_maps.append(val_map)

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train mAP: {train_map:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val mAP: {val_map:.4f} | Val Acc: {val_acc:.4f}")

        # 早停机制
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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

    # 生成并保存曲线图
    epochs = range(1, best_epoch + 1)

    # 绘制损失曲线
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses[:best_epoch], 'b-', label='Train Loss')
    plt.plot(epochs, val_losses[:best_epoch], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_curve_path = os.path.join(checkpoint_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"损失曲线已保存到 {loss_curve_path}")

    # 绘制 mAP 曲线
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_maps[:best_epoch], 'b-', label='Train mAP')
    plt.plot(epochs, val_maps[:best_epoch], 'r-', label='Val mAP')
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
