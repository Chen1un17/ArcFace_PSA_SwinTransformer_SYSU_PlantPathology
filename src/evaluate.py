# src/evaluate.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib.pyplot as plt

from dataset import PlantPathologyDataset
from model import get_model
from utils import calculate_metrics, load_checkpoint
# utils.py
from sklearn.metrics import f1_score, accuracy_score, average_precision_score

def calculate_metrics(outputs, targets, threshold=0.5):
    """
    计算 F1 分数、准确率和平均精度均值 (mAP)。

    参数:
        outputs (numpy.ndarray): 模型的输出概率，形状为 (num_samples, num_classes)
        targets (numpy.ndarray): 真实标签，形状为 (num_samples, num_classes)
        threshold (float or list): 将概率转换为二进制预测的阈值，可以是单个值或每个类别的列表

    返回:
        f1 (float): F1 分数 (macro)
        accuracy (float): 准确率
        map_score (float): 平均精度均值 (mAP) (macro)
    """
    if isinstance(threshold, list):
        preds = (outputs > np.array(threshold)).astype(int)
    else:
        preds = (outputs > threshold).astype(int)

    # 计算 F1 分数
    f1 = f1_score(targets, preds, average='macro', zero_division=0)

    # 计算准确率
    accuracy = accuracy_score(targets, preds)

    # 计算平均精度均值 (mAP)
    map_score = average_precision_score(targets, outputs, average='macro')

    return f1, accuracy, map_score

def evaluate(model, loader, criterion, device, thresholds):
    """
    评估模型在指定数据加载器上的性能，并计算各类指标。

    参数:
        model (torch.nn.Module): 已训练好的模型。
        loader (DataLoader): 数据加载器。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device): 计算设备。
        thresholds (list): 每个类别的阈值列表。

    返回:
        epoch_loss (float): 平均损失。
        metrics (dict): 各类指标，包括每个类别的 P, R, F1, AP 以及宏平均和微平均。
    """
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

    # 计算总体指标
    f1, accuracy, map_score = calculate_metrics(all_outputs, all_targets, threshold=thresholds)

    # 计算每个类别的 P, R, F1, AP
    preds = (all_outputs > np.array(thresholds)).astype(int)
    class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']

    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        all_targets, preds, average=None, zero_division=0
    )
    ap_scores = average_precision_score(all_targets, all_outputs, average=None)

    metrics = {}
    for idx, class_name in enumerate(class_names):
        metrics[class_name] = {
            'threshold': thresholds[idx],
            'P': precision[idx],
            'R': recall[idx],
            'F1': f1_scores[idx],
            'AP': ap_scores[idx]
        }

    # 计算宏平均和微平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, preds, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets, preds, average='micro', zero_division=0
    )
    ap_macro = average_precision_score(all_targets, all_outputs, average='macro')
    ap_micro = average_precision_score(all_targets, all_outputs, average='micro')

    metrics['macro-all'] = {
        'threshold': 'N/A',
        'P': precision_macro,
        'R': recall_macro,
        'F1': f1_macro,
        'AP': ap_macro
    }

    metrics['micro-all'] = {
        'threshold': 'N/A',
        'P': precision_micro,
        'R': recall_micro,
        'F1': f1_micro,
        'AP': ap_micro
    }

    return epoch_loss, metrics

def find_best_thresholds(outputs, targets, class_names, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    为每个类别在验证集上寻找最佳阈值，以最大化 F1 分数。

    参数:
        outputs (numpy.ndarray): 模型的输出概率，形状为 (num_samples, num_classes)
        targets (numpy.ndarray): 真实标签，形状为 (num_samples, num_classes)
        class_names (list): 类别名称列表
        thresholds (numpy.ndarray): 要遍历的阈值范围

    返回:
        best_thresholds (list): 每个类别的最佳阈值
        best_f1_scores (list): 每个类别对应的最佳 F1 分数
    """
    num_classes = outputs.shape[1]
    best_thresholds = []
    best_f1_scores = []

    for i in range(num_classes):
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in thresholds:
            preds = (outputs[:, i] > threshold).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)
        print(f"类别 '{class_names[i]}' 的最佳阈值: {best_threshold:.2f} | 最佳 F1 分数: {best_f1:.4f}")

    return best_thresholds, best_f1_scores

def main():
    # 配置参数
    data_dir = '/home/visllm/program/plant/Project/data'
    val_csv = os.path.join(data_dir, 'processed_val_labels.csv')  # 验证集标签
    test_csv = os.path.join(data_dir, 'processed_test_labels.csv')  # 测试集标签
    val_images = os.path.join(data_dir, 'val', 'images')
    test_images = os.path.join(data_dir, 'test', 'images')
    batch_size = 32
    num_classes = 6
    checkpoint_dir = '/home/visllm/program/plant/Project/checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 创建验证集数据加载器
    val_dataset = PlantPathologyDataset(csv_file=val_csv, images_dir=val_images, transform=transform, is_test=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 创建测试集数据加载器
    test_dataset = PlantPathologyDataset(csv_file=test_csv, images_dir=test_images, transform=transform, is_test=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=num_classes, pretrained=False)
    model = model.to(device)

    # 加载最佳模型
    model, optimizer, epoch, loss = load_checkpoint(model, None, filename=checkpoint_path)
    print(f"加载模型检查点: epoch {epoch}, loss {loss}")

    criterion = torch.nn.BCEWithLogitsLoss()

    # 在验证集上进行初步评估以获取所有输出和目标
    model.eval()
    all_val_targets = []
    all_val_outputs = []
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)
            all_val_targets.append(labels.detach().cpu().numpy())
            all_val_outputs.append(torch.sigmoid(outputs).detach().cpu().numpy())

    val_loss = running_val_loss / len(val_loader.dataset)
    all_val_targets = np.vstack(all_val_targets)
    all_val_outputs = np.vstack(all_val_outputs)

    # 定义类别名称
    class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']

    # 寻找每个类别的最佳阈值
    best_thresholds, best_f1_scores = find_best_thresholds(all_val_outputs, all_val_targets, class_names)

    # 打印验证集上的总体指标
    overall_f1, overall_accuracy, overall_map = calculate_metrics(all_val_outputs, all_val_targets, threshold=best_thresholds)
    print(f"验证集 - Loss: {val_loss:.4f} | Overall F1: {overall_f1:.4f} | Accuracy: {overall_accuracy:.4f} | mAP: {overall_map:.4f}")

    # 使用最佳阈值在验证集上进行最终评估
    final_val_loss, final_val_metrics = evaluate(model, val_loader, criterion, device, thresholds=best_thresholds)
    print(f"验证集最终评估 - Loss: {final_val_loss:.4f}")

    # 打印验证集每个类别的指标
    print("验证集每个类别的指标:")
    df_val_metrics = pd.DataFrame(final_val_metrics).T
    print(df_val_metrics)

    # 保存验证集指标到CSV
    val_metrics_df = df_val_metrics.reset_index().rename(columns={'index': 'Class'})
    val_metrics_csv_path = os.path.join(checkpoint_dir, 'validation_metrics.csv')
    val_metrics_df.to_csv(val_metrics_csv_path, index=False)
    print(f"验证集指标已保存到 {val_metrics_csv_path}")

    # 使用最佳阈值在测试集上进行评估
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, thresholds=best_thresholds)
    print(f"测试集 - Loss: {test_loss:.4f}")

    # 打印测试集每个类别的指标
    print("测试集每个类别的指标:")
    df_test_metrics = pd.DataFrame(test_metrics).T
    print(df_test_metrics)

    # 保存测试集指标到CSV
    test_metrics_df = df_test_metrics.reset_index().rename(columns={'index': 'Class'})
    test_metrics_csv_path = os.path.join(checkpoint_dir, 'test_metrics.csv')
    test_metrics_df.to_csv(test_metrics_csv_path, index=False)
    print(f"测试集指标已保存到 {test_metrics_csv_path}")

    # 绘制验证集和测试集的曲线图
    for dataset_name, metrics_df in [('validation', val_metrics_df), ('test', test_metrics_df)]:
        class_labels = metrics_df['Class']
        metrics_names = ['P', 'R', 'F1', 'AP']

        # 绘制每个指标的柱状图
        for metric in metrics_names:
            plt.figure(figsize=(10, 6))
            plt.bar(class_labels, metrics_df[metric], color='skyblue')
            plt.xlabel('Classes')
            plt.ylabel(metric)
            plt.title(f'{dataset_name.capitalize()} 集 {metric} per Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(checkpoint_dir, f'{dataset_name}_val_{metric}_per_class.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"{dataset_name.capitalize()} 集 {metric} 曲线图已保存到 {plot_path}")

        # 绘制宏平均和微平均的柱状图
        averages = ['macro-all', 'micro-all']
        for metric in metrics_names:
            plt.figure(figsize=(6, 4))
            plt.bar(averages, metrics_df[metrics_df['Class'].isin(averages)][metric], color=['orange', 'green'])
            plt.xlabel('Averages')
            plt.ylabel(metric)
            plt.title(f'{dataset_name.capitalize()} 集 {metric} Averages')
            plt.tight_layout()
            plot_path = os.path.join(checkpoint_dir, f'{dataset_name}_val_{metric}_averages.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"{dataset_name.capitalize()} 集 {metric} 平均值曲线图已保存到 {plot_path}")

        # 绘制所有指标的综合折线图
        plt.figure(figsize=(12, 8))
        for metric in metrics_names:
            plt.plot(class_labels, metrics_df[metric], marker='o', label=metric)
        plt.xlabel('Classes')
        plt.ylabel('Scores')
        plt.title(f'{dataset_name.capitalize()} 集 Metrics per Class')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        all_metrics_plot_path = os.path.join(checkpoint_dir, f'{dataset_name}_all_metrics_per_class.png')
        plt.savefig(all_metrics_plot_path)
        plt.close()
        print(f"{dataset_name.capitalize()} 集所有指标的折线图已保存到 {all_metrics_plot_path}")

if __name__ == '__main__':
    main()