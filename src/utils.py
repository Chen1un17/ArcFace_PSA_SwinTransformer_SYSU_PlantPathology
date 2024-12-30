# src/utils.py
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
import random
def calculate_metrics(outputs, targets, threshold=0.5):
    """
    计算多标签分类问题的 F1 分数、准确率和平均精度均值 (mAP)。
    
    这个函数专门处理多标签场景，其中每个样本可以同时属于多个类别。
    对于F1分数，我们分别计算每个类别的分数，然后取平均值。
    对于准确率，我们计算每个预测位置的正确率。
    对于mAP，我们使用每个类别的精确率-召回率曲线下的面积，然后取平均值。

    参数:
        outputs (numpy.ndarray): 模型的输出概率，形状为 (num_samples, num_classes)
        targets (numpy.ndarray): 真实标签，形状为 (num_samples, num_classes)
        threshold (float or list): 将概率转换为预测标签的阈值，可以是单个值或每个类别的列表

    返回:
        f1 (float): 宏平均F1分数
        accuracy (float): 样本级别的准确率
        map_score (float): 宏平均精度均值
    """
    # 将阈值转换为预测标签
    if isinstance(threshold, list):
        preds = (outputs > np.array(threshold)).astype(int)
    else:
        preds = (outputs > threshold).astype(int)
    
    # 计算宏平均F1分数：分别计算每个类别的F1分数，然后取平均值
    f1_scores = []
    for i in range(targets.shape[1]):
        class_f1 = f1_score(targets[:, i], preds[:, i], zero_division=0)
        f1_scores.append(class_f1)
    f1 = np.mean(f1_scores)
    
    # 计算多标签准确率：计算每个预测位置的正确率
    total_predictions = targets.shape[0] * targets.shape[1]  # 样本数 × 类别数
    correct_predictions = np.sum(targets == preds)  # 统计所有正确的预测位置
    accuracy = correct_predictions / total_predictions
    
    # 计算平均精度均值(mAP)：分别计算每个类别的AP，然后取平均值
    # 注意：average_precision_score已经能正确处理每个类别，所以直接使用macro平均
    map_score = average_precision_score(targets, outputs, average='macro')
    
    return f1, accuracy, map_score


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # 当optimizer不为空且checkpoint中有optimizer_state_dict才尝试加载
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, epoch, loss

import matplotlib.pyplot as plt
import torch

def tensor2img(tensor, ax=plt):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().numpy()
    return img

def subplot(images, parse=lambda x: x, rows_titles=None, cols_titles=None, title='', *args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.suptitle(title)
    i = 0
    try:
        for row in ax:
            if rows_titles is not None: row.set_title(rows_titles[i])
            try:
                for j, col in enumerate(row):
                    if cols_titles is not None:  col.set_title(cols_titles[j])
                    col.imshow(parse(images[i]))
                    col.axis('off')
                    col.set_aspect('equal')
                    i += 1
            except TypeError:
                row.imshow(parse(images[i]))
                row.axis('off')
                row.set_aspect('equal')
                i += 1
            except IndexError:
                break

    except:
        ax.imshow(parse(images[i]))
        ax.axis('off')
        ax.set_aspect('equal')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.show()


def module2traced(module, inputs):
    handles, modules = [], []

    def trace(module, inputs, outputs):
        modules.append(module)

    def traverse(module):
        for m in module.children():
            traverse(m)  # recursion is love
        is_leaf = len(list(module.children())) == 0
        if is_leaf: handles.append(module.register_forward_hook(trace))

    traverse(module)

    _ = module(inputs)

    [h.remove() for h in handles]

    return modules

def run_vis_plot(vis, x, layer, ncols=1, nrows=1):
    images, info = vis(x, layer)
    images = images[: nrows*ncols]
    print(images[0].shape)
    subplot(images, tensor2img, title=str(layer), ncols=ncols, nrows=nrows)

def run_vis_plot_across_models(modules, input, layer_id, Vis, title,
                               device,
                               inputs=None,
                               nrows=3,
                               ncols=2,
                               row_wise=True,
                               parse=tensor2img,
                               annotations=None,
                               idx2label=None,
                               rows_name=None,*args, **kwargs):
    pad = 0 # in points
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(title)

    for i, row in enumerate(ax):
        try:
            module = next(modules)
            module.eval()
            module = module.to(device)
            layer = None
            if layer_id is not None: layer = module2traced(module, input)[layer_id]
            vis = Vis(module, device)
            info = {}
            if inputs is None: images, info = vis(input.clone(), layer, *args, **kwargs)
            row_title = module.__class__.__name__
            del module
            torch.cuda.empty_cache()
            if rows_name is not None: row_title = rows_name[i]
            row[0].set_title(row_title)
            if annotations is not None:
                row[0].annotate(annotations[i], xy=(0, 0.5), xytext=(-row[0].yaxis.labelpad - pad, 0),
                    xycoords=row[0].yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center', rotation=90)
            for j, col in enumerate(row):
                if inputs is None: image = images[j]
                else: image, info = vis(inputs[j], layer, *args, **kwargs)
                if 'prediction' in info: col.set_title(idx2label[int(info['prediction'])])
                col.imshow(parse(image))
                col.axis('off')
                col.set_aspect('equal')
        except StopIteration:
            break
        except:
            row.set_title(row_title)
            row.imshow(parse(images[0]))
            row.axis('off')
            row.set_aspect('equal')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

def calculate_batch_accuracy(predictions, labels):
    """
    计算多标签分类任务中每个类别的准确率。
    
    参数:
        predictions: torch.Tensor
            模型的预测输出 (batch_size, num_classes)
            经过sigmoid激活,值域在[0,1]之间
        labels: torch.Tensor
            真实标签 (batch_size, num_classes)
            二值化的标签,只包含0和1
            
    返回:
        list: 每个类别的准确率列表
        float: 整体的平均准确率
    """
    batch_size, num_classes = predictions.size()
    
    # 将预测值和标签转换为numpy数组
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # 初始化每个类别的正确预测计数
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    # 对每个样本的每个类别分别计算
    for i in range(batch_size):
        for j in range(num_classes):
            prediction = 1 if predictions[i][j] >= 0.5 else 0
            if prediction == labels[i][j]:
                class_correct[j] += 1
            class_total[j] += 1
    
    # 计算每个类别的准确率
    class_accuracies = class_correct / class_total
    
    # 计算整体准确率(所有类别的平均)
    overall_accuracy = np.mean(class_accuracies)
    
    return class_accuracies, overall_accuracy

def set_seed(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
