import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import calculate_metrics, save_checkpoint, calculate_batch_accuracy
from dataset import PlantPathologyDataset
from model import get_model

# 实现与训练时相同的Focal Loss
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def verify_model_loading(model, checkpoint_path):
    """
    验证模型权重是否正确完整地加载
    
    参数:
        model: 要验证的模型实例
        checkpoint_path: 检查点文件路径
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    
    # 检查权重完整性
    missing_keys = set(model_dict.keys()) - set(checkpoint['model_state_dict'].keys())
    unexpected_keys = set(checkpoint['model_state_dict'].keys()) - set(model_dict.keys())
    
    if missing_keys or unexpected_keys:
        print("警告：模型加载可能不完整")
        if missing_keys:
            print("缺失的层:", missing_keys)
        if unexpected_keys:
            print("未预期的层:", unexpected_keys)
    else:
        print("模型权重加载完整，包括分类头")

def load_checkpoint(model, optimizer, filename):
    """
    加载模型检查点，适应新的保存格式
    
    参数:
        model: 要加载权重的模型
        optimizer: 优化器(如果有)
        filename: 检查点文件路径
    """
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    # 从metrics中获取loss，如果不存在则使用默认值
    loss = checkpoint['metrics']['loss']
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, epoch, loss

def tta_inference(model, image, device):
    """
    测试时增强(TTA)推理
    
    参数:
        model: 模型实例
        image: 输入图像张量
        device: 计算设备
    """
    model.eval()
    predictions = []
    
    # 原始图像预测
    with torch.no_grad():
        outputs = model(image)
        predictions.append(outputs)
    
    # 水平翻转
    img_h = torch.flip(image, dims=[3])
    outputs = model(img_h)
    predictions.append(outputs)
    
    # 垂直翻转
    img_v = torch.flip(image, dims=[2])
    outputs = model(img_v)
    predictions.append(outputs)
    
    # 90度旋转
    img_r = torch.rot90(image, k=1, dims=[2, 3])
    outputs = model(img_r)
    predictions.append(outputs)
    
    # 180度旋转
    img_r2 = torch.rot90(image, k=2, dims=[2, 3])
    outputs = model(img_r2)
    predictions.append(outputs)
    
    # 取平均得到最终预测结果
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred

def evaluate_single_model(model, loader, criterion, device, thresholds, use_tta=False):
    """
    评估单个模型的性能，提供详细的类别级别指标。
    
    这个函数实现了多个关键特性：
    1. 分别计算每个类别的准确率
    2. 支持测试时增强(TTA)以提高预测稳定性
    3. 提供全面的评估指标(精确率、召回率、F1、AP、AUC)
    4. 同时计算宏观和微观平均指标
    
    参数:
        model: 要评估的模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        thresholds: 每个类别的决策阈值
        use_tta: 是否使用测试时增强
        
    返回:
        epoch_loss: 整体损失值
        metrics: 包含所有评估指标的字典
        all_outputs: 模型在所有样本上的原始输出
        all_targets: 所有样本的真实标签
    """
    model.eval()
    running_loss = 0.0
    # 初始化每个类别的准确率统计
    running_class_accuracies = np.zeros(6)  # 假设6个类别
    num_batches = 0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 使用测试时增强或常规推理
            if use_tta:
                outputs = tta_inference(model, images, device)
            else:
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
            
            # 计算损失
            loss = criterion(outputs.to(device), labels)
            running_loss += loss.item() * images.size(0)
            
            # 计算每个类别的准确率
            batch_class_accuracies, batch_overall_accuracy = calculate_batch_accuracy(
                outputs, labels
            )
            running_class_accuracies += batch_class_accuracies
            num_batches += 1
            
            # 保存预测结果用于后续计算其他指标
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            
            # 更新进度条显示，包含每个类别的准确率
            progress_info = {
                'loss': f'{loss.item():.4f}',
                'avg_acc': f'{batch_overall_accuracy:.4f}'
            }
            for i, acc in enumerate(batch_class_accuracies):
                progress_info[f'c{i}_acc'] = f'{acc:.2f}'
            progress_bar.set_postfix(progress_info)
    
    # 计算整体损失和准确率
    epoch_loss = running_loss / len(loader.dataset)
    class_accuracies = running_class_accuracies / num_batches
    overall_accuracy = np.mean(class_accuracies)
    
    # 处理所有预测结果
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    # 定义类别名称
    class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 
                  'rust', 'complex', 'powdery_mildew']
    
    # 使用阈值进行预测
    preds = (all_outputs > np.array(thresholds)).astype(int)
    
    # 计算每个类别的详细指标
    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        all_targets, preds, average=None, zero_division=0
    )
    ap_scores = average_precision_score(all_targets, all_outputs, average=None)
    auc_scores = roc_auc_score(all_targets, all_outputs, average=None)
    
    # 构建详细的评估指标字典
    metrics = {}
    
    # 记录每个类别的指标
    for idx, class_name in enumerate(class_names):
        metrics[class_name] = {
            'threshold': thresholds[idx],
            'P': precision[idx],
            'R': recall[idx],
            'F1': f1_scores[idx],
            'AP': ap_scores[idx],
            'AUC': auc_scores[idx],
            'Accuracy': class_accuracies[idx]  # 使用类别特定的准确率
        }
    
    # 计算宏观平均指标
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, preds, average='macro', zero_division=0
    )
    ap_macro = average_precision_score(all_targets, all_outputs, average='macro')
    auc_macro = roc_auc_score(all_targets, all_outputs, average='macro')
    
    metrics['macro-avg'] = {
        'threshold': 'N/A',
        'P': precision_macro,
        'R': recall_macro,
        'F1': f1_macro,
        'AP': ap_macro,
        'AUC': auc_macro,
        'Accuracy': overall_accuracy
    }
    
    # 计算微观平均指标
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets, preds, average='micro', zero_division=0
    )
    ap_micro = average_precision_score(all_targets, all_outputs, average='micro')
    auc_micro = roc_auc_score(all_targets, all_outputs, average='micro')
    
    metrics['micro-avg'] = {
        'threshold': 'N/A',
        'P': precision_micro,
        'R': recall_micro,
        'F1': f1_micro,
        'AP': ap_micro,
        'AUC': auc_micro,
        'Accuracy': overall_accuracy
    }
    
    return epoch_loss, metrics, all_outputs, all_targets

def find_best_thresholds(outputs, targets, class_names, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    为每个类别找到最优的决策阈值
    
    参数:
        outputs: 模型输出的概率值
        targets: 真实标签
        class_names: 类别名称列表
        thresholds: 待搜索的阈值范围
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
        print(f"类别 '{class_names[i]}' - 最佳阈值: {best_threshold:.2f}, F1分数: {best_f1:.4f}")
    
    return best_thresholds, best_f1_scores

def visualize_results(metrics_df, save_dir, dataset_name):
    """
    可视化评估结果，包含各项指标的对比展示。
    
    这个函数创建三种不同的可视化：
    1. 每个指标的单独条形图
    2. 所有指标的综合对比图
    3. 指标热力图
    
    参数:
        metrics_df: 包含评估指标的DataFrame
        save_dir: 图像保存目录
        dataset_name: 数据集名称（用于图像标题）
    """
    plt.style.use('seaborn')
    
    # 定义要展示的指标
    metrics_names = ['P', 'R', 'F1', 'AP', 'AUC', 'Accuracy']
    
    # 过滤出非平均值的类别数据
    metrics_df_classes = metrics_df[~metrics_df['Class'].str.contains('-avg', na=False)].copy()
    
    # 确保所有指标列都是数值类型
    for metric in metrics_names:
        if metric in metrics_df_classes.columns:
            metrics_df_classes[metric] = pd.to_numeric(metrics_df_classes[metric], errors='coerce')
    
    # 1. 为每个指标生成单独的条形图
    for metric in metrics_names:
        if metric in metrics_df_classes.columns:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=metrics_df_classes,
                       x='Class', y=metric, 
                       palette='viridis')
            plt.title(f'{dataset_name} {metric} per Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{dataset_name}_{metric}_per_class.png'))
            plt.close()
    
    # 2. 生成所有指标的对比图
    plt.figure(figsize=(15, 8))
    for metric in metrics_names:
        if metric in metrics_df_classes.columns:
            plt.plot(metrics_df_classes['Class'], 
                    metrics_df_classes[metric],
                    marker='o', 
                    label=metric)
    
    plt.title(f'{dataset_name} All Metrics Comparison')
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_all_metrics.png'))
    plt.close()
    
    # 3. 生成热力图
    # 只选择数值列并确保数据类型正确
    numeric_metrics = metrics_df_classes[metrics_names].astype(float)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_metrics, 
                annot=True, 
                fmt='.3f',
                xticklabels=metrics_names,
                yticklabels=metrics_df_classes['Class'],
                cmap='YlOrRd')
    plt.title(f'{dataset_name} Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_metrics_heatmap.png'))
    plt.close()

def main():
    """
    主函数：执行完整的评估流程
    包括数据加载、模型初始化、阈值优化、性能评估和结果可视化
    """
    # 配置参数与路径
    data_dir = '/home/visllm/program/plant/Project/data'
    val_csv = os.path.join(data_dir, 'processed_val_labels.csv')
    test_csv = os.path.join(data_dir, 'processed_test_labels.csv')
    val_images = os.path.join(data_dir, 'val', 'images')
    test_images = os.path.join(data_dir, 'test', 'images')
    
    batch_size = 32
    num_classes = 6
    model_name = 'swinv2'
    checkpoint_dir = '/home/visllm/program/plant/Project/checkpoints_swin_enhanced_train'
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # 数据预处理转换
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集和数据加载器
    print("\n创建数据加载器...")
    val_dataset = PlantPathologyDataset(
        csv_file=val_csv,
        images_dir=val_images,
        transform=transform,
        is_test=True
    )
    test_dataset = PlantPathologyDataset(
        csv_file=test_csv,
        images_dir=test_images,
        transform=transform,
        is_test=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 设备配置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("\n初始化模型...")
    model = get_model(
        num_classes=num_classes,
        pretrained=True,
        model_name=model_name
    )
    model = model.to(device)
    
    # 使用与训练相同的Focal Loss损失函数
    criterion = FocalBCELoss(gamma=2, alpha=0.25)
    
    # 加载模型权重并验证
    print("\n加载模型权重...")
    model, _, epoch, loss = load_checkpoint(model, None, checkpoint_path)
    verify_model_loading(model, checkpoint_path)
    print(f"已加载模型检查点: epoch {epoch}, loss {loss:.4f}")
    
    # 在验证集上找到最佳阈值
    print("\n在验证集上评估并找到最佳阈值...")
    val_loss, val_metrics_initial, val_outputs, val_targets = evaluate_single_model(
        model,
        val_loader,
        criterion,
        device,
        thresholds=[0.5] * num_classes,  # 初始使用0.5作为所有类别的阈值
        use_tta=True
    )
    
    # 获取类别名称列表
    class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
    
    # 为每个类别找到最佳阈值
    print("\n优化每个类别的决策阈值...")
    best_thresholds, best_f1_scores = find_best_thresholds(
        val_outputs,
        val_targets,
        class_names
    )
    
    # 使用最佳阈值进行最终验证集评估
    print("\n使用优化后的阈值进行验证集最终评估...")
    final_val_loss, final_val_metrics, _, _ = evaluate_single_model(
        model,
        val_loader,
        criterion,
        device,
        thresholds=best_thresholds,
        use_tta=True
    )
    
    # 打印验证集评估结果
    print("\n验证集最终评估结果:")
    df_val_metrics = pd.DataFrame(final_val_metrics).T
    print(df_val_metrics)
    
    # 在测试集上评估
    print("\n使用优化后的阈值在测试集上评估...")
    test_loss, test_metrics, _, _ = evaluate_single_model(
        model,
        test_loader,
        criterion,
        device,
        thresholds=best_thresholds,
        use_tta=True
    )
    
    # 打印测试集评估结果
    print("\n测试集评估结果:")
    df_test_metrics = pd.DataFrame(test_metrics).T
    print(df_test_metrics)
    
    # 保存评估结果到CSV文件
    val_metrics_df = df_val_metrics.reset_index().rename(columns={'index': 'Class'})
    test_metrics_df = df_test_metrics.reset_index().rename(columns={'index': 'Class'})
    
    val_metrics_csv_path = os.path.join(checkpoint_dir, 'validation_metrics.csv')
    test_metrics_csv_path = os.path.join(checkpoint_dir, 'test_metrics.csv')
    
    val_metrics_df.to_csv(val_metrics_csv_path, index=False)
    test_metrics_df.to_csv(test_metrics_csv_path, index=False)
    
    print(f"\n验证集指标已保存到: {val_metrics_csv_path}")
    print(f"测试集指标已保存到: {test_metrics_csv_path}")
    
    # 生成和保存可视化结果
    print("\n生成评估结果可视化...")
    visualize_results(val_metrics_df, checkpoint_dir, 'Validation')
    visualize_results(test_metrics_df, checkpoint_dir, 'Test')
    
    # 保存优化后的阈值
    thresholds_df = pd.DataFrame({
        'Class': class_names,
        'Optimal_Threshold': best_thresholds,
        'Best_F1_Score': best_f1_scores
    })
    thresholds_path = os.path.join(checkpoint_dir, 'optimal_thresholds.csv')
    thresholds_df.to_csv(thresholds_path, index=False)
    print(f"\n最优阈值已保存到: {thresholds_path}")
    
    # 打印最终性能总结
    print("\n性能评估总结:")
    print("-" * 50)
    print("验证集性能:")
    print(f"Macro-Avg F1: {final_val_metrics['macro-avg']['F1']:.4f}")
    print(f"Micro-Avg F1: {final_val_metrics['micro-avg']['F1']:.4f}")
    print(f"平均 AP: {final_val_metrics['macro-avg']['AP']:.4f}")
    print(f"平均 AUC: {final_val_metrics['macro-avg']['AUC']:.4f}")
    print("\n测试集性能:")
    print(f"Macro-Avg F1: {test_metrics['macro-avg']['F1']:.4f}")
    print(f"Micro-Avg F1: {test_metrics['micro-avg']['F1']:.4f}")
    print(f"平均 AP: {test_metrics['macro-avg']['AP']:.4f}")
    print(f"平均 AUC: {test_metrics['macro-avg']['AUC']:.4f}")
    print("-" * 50)
    
    print("\n评估完成！所有结果已保存到:", checkpoint_dir)

if __name__ == '__main__':
    main()