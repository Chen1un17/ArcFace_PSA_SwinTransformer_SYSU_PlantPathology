# src/evaluate.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score

from dataset import PlantPathologyDataset
from model import get_model
from utils import calculate_metrics, load_checkpoint

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in loader:
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
    f1, accuracy = calculate_metrics(all_outputs, all_targets)
    return epoch_loss, f1, accuracy

def main():
    # 配置参数
    data_dir = '/home/visllm/program/plant/Project/data'
    val_csv = os.path.join(data_dir, 'processed_test_labels.csv')
    val_images = os.path.join(data_dir, 'test', 'images')
    batch_size = 32
    num_classes = 6
    checkpoint_dir = '/home/visllm/program/plant/Project/checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # 数据预处理
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和数据加载器
    val_dataset = PlantPathologyDataset(csv_file=val_csv, images_dir=val_images, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    device = torch.device("cuda:1")
    model = get_model(num_classes=num_classes, pretrained=False)
    model = model.to(device)

    # 加载最佳模型
    model, optimizer, epoch, loss = load_checkpoint(model, None, filename=checkpoint_path)
    print(f"加载模型检查点: epoch {epoch}, loss {loss}")

    criterion = torch.nn.BCEWithLogitsLoss()

    # 评估
    val_loss, val_f1, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"验证集 - Loss: {val_loss:.4f} | F1: {val_f1:.4f} | Accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    main()
