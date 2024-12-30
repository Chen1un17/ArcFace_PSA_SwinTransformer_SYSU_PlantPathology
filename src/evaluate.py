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

# 保留你项目中的辅助函数
from utils import calculate_metrics, save_checkpoint, calculate_batch_accuracy
from dataset import PlantPathologyDataset
from model import get_model, PSANArcFaceModel
from train import tta_inference  # 仍然沿用 train.py 的 TTA 推断
from lossfunction import MultiLabelArcFaceLoss, FocalBCELoss


def verify_model_loading(model, checkpoint_path):
    """
    验证模型权重是否正确完整地加载
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    
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
    """
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['metrics']['loss']
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, epoch, loss


def evaluate_single_model(model, loader, criterion, device, thresholds, use_tta=False):
    """
    评估单个模型的性能，尽量与 train.py 中的 eval_epoch 保持一致的逻辑和返回结构。

    返回:
      epoch_loss: float, 当前数据集的平均loss
      metrics: dict，包含:
        {
          'loss': epoch_loss,
          'accuracy': overall_accuracy,
          'class_accuracies': list,  # 每个类别准确率
          'f1': f1_score,            # 由 calculate_metrics 返回
          'map': map_score           # 由 calculate_metrics 返回
          # 另外保留 macro-avg, micro-avg 字段供外部可视化或打印 (如果需要)
        }
      all_outputs: (N, C) 的sigmoid后概率
      all_targets: (N, C) 的真实标签
    """
    model.eval()
    running_loss = 0.0
    running_class_accuracies = np.zeros(6)  # 6个类别
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
    
    # 总样本数
    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    
    all_targets = np.vstack(all_targets)  # (N, 6)
    all_outputs = np.vstack(all_outputs)  # (N, 6)
    
    # 利用与 train.py 相同的 calculate_metrics => (f1, accuracy, map)
    # 这里 accuracy 在 calculate_metrics 中是逐标签水平的准确率，与 batch_class_accuracies 含义类似
    # 为了与train.py保持一致，这里依然采用 f1、map即可
    f1, _, map_score = calculate_metrics(all_outputs, all_targets)
    
    # 计算每个类别和整体准确率（与train.py中eval_epoch保持一致）
    class_accuracies = running_class_accuracies / num_batches
    overall_accuracy = np.mean(class_accuracies)
    
    # ------- 如果你还想保留宏观/微观平均等更详细指标，可继续保留 -------
    # 根据传进来的 thresholds 做最终二值化预测
    preds = (all_outputs > np.array(thresholds)).astype(int)
    
    # 这里可选：计算 macro/micro 指标
    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        all_targets, preds, average=None, zero_division=0
    )
    ap_scores = average_precision_score(all_targets, all_outputs, average=None)
    auc_scores = roc_auc_score(all_targets, all_outputs, average=None)
    
    # macro
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, preds, average='macro', zero_division=0
    )
    ap_macro = average_precision_score(all_targets, all_outputs, average='macro')
    auc_macro = roc_auc_score(all_targets, all_outputs, average='macro')
    
    # micro
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets, preds, average='micro', zero_division=0
    )
    ap_micro = average_precision_score(all_targets, all_outputs, average='micro')
    auc_micro = roc_auc_score(all_targets, all_outputs, average='micro')
    
    # 最终返回的指标，与 train.py 中eval_epoch风格保持一致
    metrics = {
        'loss': epoch_loss,
        'accuracy': overall_accuracy,
        'class_accuracies': class_accuracies.tolist(),
        'f1': f1,         # 由 calculate_metrics() 返回的多标签F1(宏平均)
        'map': map_score, # 由 calculate_metrics() 返回的mAP(宏平均)
        
        # 如果需要外部可视化 micro/macro 结果，可加在这里:
        'macro-avg': {
            'F1': f1_macro,
            'AP': ap_macro,
            'AUC': auc_macro,
            'Accuracy': overall_accuracy
        },
        'micro-avg': {
            'F1': f1_micro,
            'AP': ap_micro,
            'AUC': auc_micro,
            'Accuracy': overall_accuracy
        }
    }
    
    return epoch_loss, metrics, all_outputs, all_targets


def find_best_thresholds(outputs, targets, class_names, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    为每个类别找到最优阈值(使F1最大)
    """
    num_classes = outputs.shape[1]
    best_thresholds = []
    best_f1_scores = []
    
    for i in range(num_classes):
        best_threshold = 0.5
        best_f1 = 0.0
        
        for thr in thresholds:
            preds = (outputs[:, i] > thr).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr
        
        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)
        print(f"类别 '{class_names[i]}' - 最佳阈值: {best_threshold:.2f}, F1分数: {best_f1:.4f}")
    
    return best_thresholds, best_f1_scores


def visualize_results(metrics_df, save_dir, dataset_name):
    """
    可视化评估结果
    """
    plt.style.use('seaborn')
    metrics_names = ['P', 'R', 'F1', 'AP', 'AUC', 'Accuracy']
    
    metrics_df_classes = metrics_df[~metrics_df['Class'].str.contains('-avg', na=False)].copy()
    for metric in metrics_names:
        if metric in metrics_df_classes.columns:
            metrics_df_classes[metric] = pd.to_numeric(metrics_df_classes[metric], errors='coerce')
    
    # 1) 为每个指标生成单独的条形图
    for metric in metrics_names:
        if metric in metrics_df_classes.columns:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=metrics_df_classes, x='Class', y=metric, palette='viridis')
            plt.title(f'{dataset_name} {metric} per Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{dataset_name}_{metric}_per_class.png'))
            plt.close()
    
    # 2) 生成所有指标的折线对比图
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
    
    # 3) 生成热力图
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
    """
    # 路径配置
    data_dir = '/home/visllm/program/plant/Project/data'
    val_csv = os.path.join(data_dir, 'processed_val_labels.csv')
    test_csv = os.path.join(data_dir, 'processed_test_labels.csv')
    val_images = os.path.join(data_dir, 'val', 'images')
    test_images = os.path.join(data_dir, 'test', 'images')
    
    batch_size = 32
    num_classes = 6
    model_name = 'swinv2'
    checkpoint_dir = '/home/visllm/program/plant/Project/checkpoints_swin_enhanced_arcface'
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # 数据集 & DataLoader
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
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("\n初始化模型...")
    model = get_model(
        num_classes=num_classes,
        pretrained=True,
        model_name=model_name,
        use_arcface=True,
        feature_dim=512
    )
    model = model.to(device)
    
    # 根据模型类型选择损失函数
    if isinstance(model, PSANArcFaceModel):
        criterion = MultiLabelArcFaceLoss(
            in_features=512,
            num_classes=num_classes,
            scale=30.0,
            margin=0.5
        ).to(device)
    else:
        criterion = FocalBCELoss(gamma=2, alpha=0.25)
    
    # 加载checkpoint并验证
    print("\n加载模型权重...")
    model, _, epoch, loss = load_checkpoint(model, None, checkpoint_path)
    verify_model_loading(model, checkpoint_path)
    print(f"已加载模型检查点: epoch {epoch}, loss {loss:.4f}")
    
    # 在验证集上先用阈值=0.5评估
    print("\n在验证集上评估并找到最佳阈值...")
    val_loss, val_metrics_initial, val_outputs, val_targets = evaluate_single_model(
        model,
        val_loader,
        criterion,
        device,
        thresholds=[0.5]*num_classes,
        use_tta=True
    )
    
    class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
    
    # 优化阈值
    print("\n优化每个类别的决策阈值...")
    best_thresholds, best_f1_scores = find_best_thresholds(
        val_outputs,
        val_targets,
        class_names
    )
    
    # 在验证集上用最优阈值重新评估
    print("\n使用优化后的阈值进行验证集最终评估...")
    final_val_loss, final_val_metrics, _, _ = evaluate_single_model(
        model,
        val_loader,
        criterion,
        device,
        thresholds=best_thresholds,
        use_tta=True
    )
    
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
    
    print("\n测试集评估结果:")
    df_test_metrics = pd.DataFrame(test_metrics).T
    print(df_test_metrics)
    
    # 保存结果
    val_metrics_df = df_val_metrics.reset_index().rename(columns={'index': 'Class'})
    test_metrics_df = df_test_metrics.reset_index().rename(columns={'index': 'Class'})
    
    val_metrics_csv_path = os.path.join(checkpoint_dir, 'validation_metrics.csv')
    test_metrics_csv_path = os.path.join(checkpoint_dir, 'test_metrics.csv')
    val_metrics_df.to_csv(val_metrics_csv_path, index=False)
    test_metrics_df.to_csv(test_metrics_csv_path, index=False)
    print(f"\n验证集指标已保存到: {val_metrics_csv_path}")
    print(f"测试集指标已保存到: {test_metrics_csv_path}")
    
    # 可视化
    print("\n生成评估结果可视化...")
    visualize_results(val_metrics_df, checkpoint_dir, 'Validation')
    visualize_results(test_metrics_df, checkpoint_dir, 'Test')
    
    # 保存最优阈值
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
    print(f"F1 (from train.py style): {final_val_metrics['f1']:.4f}")
    print(f"mAP: {final_val_metrics['map']:.4f}")
    print(f"Accuracy (pos-level): {final_val_metrics['accuracy']:.4f}")
    
    if 'macro-avg' in final_val_metrics:
        print(f"Macro-Avg F1: {final_val_metrics['macro-avg']['F1']:.4f}")
        print(f"Macro-Avg AP: {final_val_metrics['macro-avg']['AP']:.4f}")
        print(f"Macro-Avg AUC: {final_val_metrics['macro-avg']['AUC']:.4f}")
    if 'micro-avg' in final_val_metrics:
        print(f"Micro-Avg F1: {final_val_metrics['micro-avg']['F1']:.4f}")
        print(f"Micro-Avg AP: {final_val_metrics['micro-avg']['AP']:.4f}")
        print(f"Micro-Avg AUC: {final_val_metrics['micro-avg']['AUC']:.4f}")
    
    print("\n测试集性能:")
    print(f"F1 (from train.py style): {test_metrics['f1']:.4f}")
    print(f"mAP: {test_metrics['map']:.4f}")
    print(f"Accuracy (pos-level): {test_metrics['accuracy']:.4f}")
    
    if 'macro-avg' in test_metrics:
        print(f"Macro-Avg F1: {test_metrics['macro-avg']['F1']:.4f}")
        print(f"Macro-Avg AP: {test_metrics['macro-avg']['AP']:.4f}")
        print(f"Macro-Avg AUC: {test_metrics['macro-avg']['AUC']:.4f}")
    if 'micro-avg' in test_metrics:
        print(f"Micro-Avg F1: {test_metrics['micro-avg']['F1']:.4f}")
        print(f"Micro-Avg AP: {test_metrics['micro-avg']['AP']:.4f}")
        print(f"Micro-Avg AUC: {test_metrics['micro-avg']['AUC']:.4f}")
    
    print("-" * 50)
    
    print("\n评估完成！所有结果已保存到:", checkpoint_dir)


if __name__ == '__main__':
    main()
