# src/generate_submission.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PlantPathologyDataset
from model import get_model
from utils import load_checkpoint

def main():
    # 配置参数
    data_dir = '/home/visllm/program/plant/Project/data'
    test_csv = os.path.join(data_dir, 'processed_test_labels.csv')  
    test_images = os.path.join(data_dir, 'test', 'images')
    batch_size = 32
    num_classes = 6
    checkpoint_dir = os.path.join(data_dir, '/home/visllm/program/plant/Project/checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    submission_dir = os.path.join(data_dir, '/home/visllm/program/plant/Project/submissions')
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, 'submission.csv')

    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和数据加载器，设置 is_test=True
    test_dataset = PlantPathologyDataset(csv_file=test_csv, images_dir=test_images, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    device = torch.device("cuda:3")
    model = get_model(num_classes=num_classes, pretrained=False)
    model = model.to(device)

    # 加载最佳模型
    model, optimizer, epoch, loss = load_checkpoint(model, None, filename=checkpoint_path)
    print(f"加载模型检查点: epoch {epoch}, loss {loss}")

    model.eval()
    all_outputs = []
    image_names = []

    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_outputs.append(probs)
            image_names.extend(img_names)

    all_outputs = np.vstack(all_outputs)

    # 生成二进制标签
    pred_labels = (all_outputs > 0.5).astype(int)

    # 创建提交文件
    submission = pd.DataFrame(pred_labels, columns=['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew'])
    submission.insert(0, 'images', image_names)

    # 保存为 CSV 文件
    submission.to_csv(submission_path, index=False)
    print(f"测试结果已保存为 {submission_path}")

if __name__ == '__main__':
    main()
