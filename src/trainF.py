import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

from dataset import PlantPathologyDataset
from model import get_model  
from utils import (
    calculate_metrics,
    save_checkpoint,
    calculate_batch_accuracy,
    set_seed
)
from lossfunction import FocalBCELoss, MultiLabelArcFaceLoss

# ============ 可视化评估函数 ============
def visualize_results(metrics_df, save_dir, dataset_name):
    plt.style.use('seaborn')
    metrics_names = ['P', 'R', 'F1', 'AP', 'AUC', 'Accuracy']
    
    metrics_df_classes = metrics_df[~metrics_df['Class'].str.contains('-avg', na=False)].copy()
    
    for metric in metrics_names:
        if metric in metrics_df_classes.columns:
            metrics_df_classes[metric] = pd.to_numeric(metrics_df_classes[metric], errors='coerce')
    
    # (1) 为每个指标生成单独的条形图
    for metric in metrics_names:
        if metric in metrics_df_classes.columns:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=metrics_df_classes, x='Class', y=metric, palette='viridis')
            plt.title(f'{dataset_name} {metric} per Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{dataset_name}_{metric}_per_class.png'))
            plt.close()
    
    # (2) 生成所有指标的折线对比图
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
    
    # (3) 生成热力图
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

# ============ 测试时增强(TTA)示例(可选) ============
def tta_inference(model, images, device):
    model.eval()
    features_list = []
    
    transforms_list = [
        lambda x: x,  # 原图
        lambda x: torch.flip(x, dims=[3]),  # 水平翻转
        lambda x: torch.flip(x, dims=[2]),  # 垂直翻转
        lambda x: torch.rot90(x, k=1, dims=[2, 3])  # 旋转90度
    ]
    
    with torch.no_grad():
        for fn in transforms_list:
            img_t = fn(images)
            feats = model(img_t)  # 这里默认 model 输出 logits 或特征，需按实际情况改写
            features_list.append(feats)
    
    return torch.stack(features_list).mean(dim=0)

def get_transforms(resize_size, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            RandAugment(num_ops=2, magnitude=9),
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

# ============ 评估(含逐类别指标)的函数 ============
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score
)

def evaluate_with_details(model, loader, criterion, device, thresholds, use_tta=False):
    model.eval()
    running_loss = 0.0
    running_class_accuracies = np.zeros(6)
    num_batches = 0
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if use_tta:
                features = tta_inference(model, images, device)
            else:
                features = model(images)
            
            # 如果是ArcFace，需要再用criterion(features, labels)拿到 logits:
            loss, logits = criterion(features, labels)  # (loss, logits)
            probs = torch.sigmoid(logits)
            
            batch_class_accuracies, batch_acc = calculate_batch_accuracy(probs, labels)
            running_class_accuracies += batch_class_accuracies
            num_batches += 1
            
            running_loss += loss.item() * images.size(0)
            
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
    
    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    
    all_targets = np.vstack(all_targets)  # (N,6)
    all_outputs = np.vstack(all_outputs)  # (N,6)
    
    # 由 utils.calculate_metrics => (f1, accuracy, map)
    f1_val, _, map_val = calculate_metrics(all_outputs, all_targets)
    
    class_accuracies = running_class_accuracies / num_batches
    overall_accuracy = np.mean(class_accuracies)
    
    # 根据传入 thresholds 做二值化
    preds = (all_outputs > np.array(thresholds)).astype(int)
    
    # 每个类别的PRF
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        all_targets, preds, average=None, zero_division=0
    )
    ap_arr = average_precision_score(all_targets, all_outputs, average=None)
    auc_arr = roc_auc_score(all_targets, all_outputs, average=None)
    
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
    
    # 整体结果
    overall_metrics = {
        'loss': epoch_loss,
        'accuracy': overall_accuracy,  # pos-level
        'class_accuracies': class_accuracies.tolist(),
        'f1': f1_val,   # 由 calculate_metrics 返回(宏平均)
        'map': map_val, # 由 calculate_metrics 返回(宏平均)
        
        'macro-avg': {
            'F1': f1_macro,
            'AP': ap_macro,
            'AUC': auc_macro,
            'Accuracy': overall_accuracy  # 这里保留pos-level
        },
        'micro-avg': {
            'F1': f1_micro,
            'AP': ap_micro,
            'AUC': auc_micro,
            'Accuracy': overall_accuracy
        }
    }
    
    # 构建逐类别的 DataFrame
    class_names = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
    per_class_data = []
    for i, cname in enumerate(class_names):
        per_class_data.append({
            'Class': cname,
            'P': precision_arr[i],
            'R': recall_arr[i],
            'F1': f1_arr[i],
            'AP': ap_arr[i],
            'AUC': auc_arr[i],
            'Accuracy': class_accuracies[i]
        })
    
    # 也可以加一行 macro-avg, micro-avg
    per_class_data.append({
        'Class': 'macro-avg',
        'P': precision_macro,
        'R': recall_macro,
        'F1': f1_macro,
        'AP': ap_macro,
        'AUC': auc_macro,
        'Accuracy': overall_accuracy
    })
    per_class_data.append({
        'Class': 'micro-avg',
        'P': precision_micro,
        'R': recall_micro,
        'F1': f1_micro,
        'AP': ap_micro,
        'AUC': auc_micro,
        'Accuracy': overall_accuracy
    })
    
    per_class_df = pd.DataFrame(per_class_data)
    
    return overall_metrics, per_class_df

def train_epoch(model, loader, criterion, optimizer, device, scaler, accumulation_steps=2):
    """
    训练一个epoch
    """
    model.train()
    running_loss = 0.0
    running_class_accuracies = np.zeros(6)
    num_batches = 0
    
    all_targets = []
    all_outputs = []
    
    optimizer.zero_grad()
    progress_bar = tqdm(loader, desc="Training")
    
    for idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        with autocast():
            features = model(images)
            loss, logits = criterion(features, labels)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        probs = torch.sigmoid(logits)
        batch_class_accuracies, batch_acc = calculate_batch_accuracy(probs, labels)
        running_class_accuracies += batch_class_accuracies
        num_batches += 1
        
        running_loss += loss.item() * accumulation_steps * images.size(0)
        
        all_targets.append(labels.detach().cpu().numpy())
        all_outputs.append(probs.detach().cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{batch_acc:.4f}'
        })
    
    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    all_targets = np.vstack(all_targets)  # (N,6)
    all_outputs = np.vstack(all_outputs)  # (N,6)
    
    f1_val, _, map_val = calculate_metrics(all_outputs, all_targets)
    
    class_accuracies = running_class_accuracies / num_batches
    overall_accuracy = np.mean(class_accuracies)
    
    return {
        'loss': epoch_loss,
        'accuracy': overall_accuracy,
        'class_accuracies': class_accuracies.tolist(),
        'f1': f1_val,
        'map': map_val
    }

# ============ 新增: 在验证集上搜索最佳阈值 ============
from sklearn.metrics import f1_score

def find_optimal_thresholds(model, loader, device, criterion, num_classes=6, step=0.01, use_tta=False):
    model.eval()
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if use_tta:
                features = tta_inference(model, images, device)
            else:
                features = model(images)
            
            _, logits = criterion(features, labels)
            probs = torch.sigmoid(logits)
            
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
    
    all_targets = np.vstack(all_targets)   # (N, num_classes)
    all_outputs = np.vstack(all_outputs)   # (N, num_classes)
    
    thresholds = [0.5] * num_classes
    search_range = np.arange(0.0, 1.0 + 1e-9, step)  # step=0.01
    
    for c in range(num_classes):
        best_thr = 0.5
        best_f1 = 0.0
        y_true = all_targets[:, c]
        y_prob = all_outputs[:, c]
        
        for thr in search_range:
            y_pred = (y_prob >= thr).astype(int)
            f1_c = f1_score(y_true, y_pred, average='binary', zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_thr = thr
        
        thresholds[c] = best_thr
    
    return thresholds

# ============ 新增：eval_epoch 用于验证或测试 ============

def eval_epoch(model, loader, criterion, device):
    """
    验证/测试用的单独评估函数
    （与 train_epoch 类似，但不做反向传播、不更新参数）
    返回的指标格式与 train_epoch 保持一致，方便比较和记录
    """
    model.eval()
    running_loss = 0.0
    running_class_accuracies = np.zeros(6)
    num_batches = 0
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # forward
            features = model(images)
            loss, logits = criterion(features, labels)
            
            probs = torch.sigmoid(logits)
            batch_class_accuracies, batch_acc = calculate_batch_accuracy(probs, labels)
            running_class_accuracies += batch_class_accuracies
            num_batches += 1
            
            running_loss += loss.item() * images.size(0)
            
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
    
    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    
    all_targets = np.vstack(all_targets)  # (N,6)
    all_outputs = np.vstack(all_outputs)  # (N,6)
    
    f1_val, _, map_val = calculate_metrics(all_outputs, all_targets)
    
    class_accuracies = running_class_accuracies / num_batches
    overall_accuracy = np.mean(class_accuracies)
    
    return {
        'loss': epoch_loss,
        'accuracy': overall_accuracy,
        'class_accuracies': class_accuracies.tolist(),
        'f1': f1_val,
        'map': map_val
    }

def main():
    set_seed(42)
    
    # 路径配置
    data_dir = '/home/visllm/program/plant/Project/data'
    train_csv = os.path.join(data_dir, 'processed_train_labels.csv')
    val_csv   = os.path.join(data_dir, 'processed_val_labels.csv')
    test_csv  = os.path.join(data_dir, 'processed_test_labels.csv')
    
    train_images = os.path.join(data_dir, 'train', 'images')
    val_images   = os.path.join(data_dir, 'val', 'images')
    test_images  = os.path.join(data_dir, 'test', 'images')
    
    # 训练超参数
    batch_size = 16
    num_epochs = 15   
    learning_rate = 1e-4
    num_classes = 6
    accumulation_steps = 2
    resize_size = 384
    
    # 创建实验目录
    checkpoint_dir = os.path.join(data_dir, '../checkpoints_swin_enhanced_arcface_PSAN')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建数据集
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
    test_dataset = PlantPathologyDataset(
        csv_file=test_csv,
        images_dir=test_images,
        transform=get_transforms(resize_size, is_training=False),
        is_test=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型和相关组件
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = get_model(
        num_classes=num_classes,
        pretrained=True,
        model_name='swinv2',
        use_arcface=True,  # ArcFace
        feature_dim=512
    )
    model.to(device)
    criterion = MultiLabelArcFaceLoss(
        in_features=512,
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
    
    best_val_f1 = 0.0
    patience = 5
    counter = 0
    
    history = defaultdict(list)
    
    # ============ 训练循环 ============
    for epoch in range(1, num_epochs+1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        
        # 1) 训练
        train_metrics = train_epoch(
            model, train_loader, criterion,
            optimizer, device, scaler,
            accumulation_steps
        )
        scheduler.step()
        
        # 2) 验证：使用单独的 eval_epoch
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        
        # 3) 记录
        for phase, m in [('train', train_metrics), ('val', val_metrics)]:
            for k, v in m.items():
                history[f'{phase}_{k}'].append(v)
        
        # 打印
        print(f"[Train] loss={train_metrics['loss']:.4f}, f1={train_metrics['f1']:.4f}, "
              f"acc={train_metrics['accuracy']:.4f}, map={train_metrics['map']:.4f}")
        print(f"[Val]   loss={val_metrics['loss']:.4f}, f1={val_metrics['f1']:.4f}, "
              f"acc={val_metrics['accuracy']:.4f}, map={val_metrics['map']:.4f}")
        
        # 4) Early stopping & 保存最好模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            counter = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'best_val_f1': best_val_f1,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print("Saved best model.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
    # ============ 训练结束 ============
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(checkpoint_dir, 'train_history.csv'), index=False)
    print("\nTraining completed. History saved.")
    
    # ========== 在验证集上搜索最佳阈值 ==========
    print("\n开始在验证集上搜索最优阈值...")
    best_thresholds = find_optimal_thresholds(
        model,
        val_loader,
        device,
        criterion,
        num_classes=num_classes,
        step=0.01,
        use_tta=False
    )
    print("在验证集上搜索到的最佳阈值: ", best_thresholds)

    # ====== 在验证集 & 测试集做最终评估 ======
    final_val_metrics, final_val_df = evaluate_with_details(
        model, val_loader, criterion, device, thresholds=best_thresholds, use_tta=False
    )
    final_val_df.to_csv(os.path.join(checkpoint_dir, 'validation_details.csv'), index=False)
    visualize_results(final_val_df, checkpoint_dir, 'Validation')
    
    test_metrics, test_df = evaluate_with_details(
        model, test_loader, criterion, device, thresholds=best_thresholds, use_tta=False
    )
    test_df.to_csv(os.path.join(checkpoint_dir, 'test_details.csv'), index=False)
    visualize_results(test_df, checkpoint_dir, 'Test')
    
    # 最后打印性能总结
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
    
    print("\n全部评估完成，结果已输出到：", checkpoint_dir)


if __name__ == '__main__':
    main()
