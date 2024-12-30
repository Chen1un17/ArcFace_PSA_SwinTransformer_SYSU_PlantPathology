import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PlantPathologyDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, is_test=False):
        """
        初始化数据集
        
        参数:
            csv_file (str): 包含图像名称和标签的CSV文件路径
            images_dir (str): 图像文件夹路径
            transform: 图像转换操作
            is_test (bool): 是否为测试模式
        """
        self.original_frame = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.is_test = is_test
        
        # 训练模式下对complex类进行数据增强
        self.labels_frame = self._augment_complex_class() if not is_test else self.original_frame

    def _augment_complex_class(self):
        """
        专门针对complex类别进行数据增强
        
        增强策略:
        1. 找出所有包含complex标签的样本
        2. 将这些样本重复5次以平衡数据集
        3. 将增强后的样本与原始数据集合并
        4. 随机打乱最终数据集
        
        返回:
            pd.DataFrame: 增强后的数据框
        """
        # 获取complex类的样本数量
        complex_count = self.original_frame['complex'].sum()
        print(f"Original complex samples: {complex_count}")
        
        # 提取complex类的样本
        complex_samples = self.original_frame[self.original_frame['complex'] == 1]
        
        # 如果没有complex样本，直接返回原始数据集
        if len(complex_samples) == 0:
            print("No complex samples found in the dataset")
            return self.original_frame
        
        # 创建增强数据列表
        augmented_frames = [self.original_frame]  # 首先添加原始数据
        
        # 重复complex样本5次
        for _ in range(5):
            augmented_frames.append(complex_samples.copy())
        
        # 合并所有数据框
        final_frame = pd.concat(augmented_frames, ignore_index=True)
        
        # 随机打乱数据顺序
        final_frame = final_frame.sample(frac=1).reset_index(drop=True)
        
        # 打印增强后的complex样本数量
        final_complex_count = final_frame['complex'].sum()
        print(f"Augmented complex samples: {final_complex_count}")
        
        return final_frame

    def __len__(self):
        """返回数据集大小"""
        return len(self.labels_frame)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        参数:
            idx (int): 样本索引
        
        返回:
            tuple: (转换后的图像, 标签)
        """
        # 读取图像
        img_name = os.path.join(self.images_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # 应用图像转换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        labels = self.labels_frame.iloc[idx, 1:].values.astype('float')
        
        return image, labels