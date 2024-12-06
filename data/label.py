import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# 定义 train_label.csv 文件的路径
csv_path = '/home/visllm/program/plant/plant_dataset/val/val_label.csv'  # 请替换为您的实际路径

# 读取 CSV 文件
df = pd.read_csv(csv_path)
print('原始数据框：')
print(df.head())

# 定义六个类别
classes = [
    'scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex',
    'powdery_mildew'
]


# 定义函数，将标签字符串拆分为列表
def split_labels(label_str):
    # 以空格为分隔符拆分标签
    return label_str.split()


# 应用函数，创建一个新的列，包含标签列表
df['labels_list'] = df['labels'].apply(split_labels)

print('\n处理后的数据框：')
print(df.head())

# 使用 MultiLabelBinarizer 进行 One-Hot 编码
mlb = MultiLabelBinarizer(classes=classes)

# 拟合并转换标签列表
labels_one_hot = mlb.fit_transform(df['labels_list'])

# 将结果转换回 DataFrame，便于查看
labels_one_hot_df = pd.DataFrame(labels_one_hot, columns=mlb.classes_)

print('\nOne-Hot 编码的标签：')
print(labels_one_hot_df.head())

# 将图片名称和编码后的标签合并
final_df = pd.concat([df['images'], labels_one_hot_df], axis=1)

print('\n最终的数据框：')
print(final_df.head())

# 如果需要，可以将处理后的数据保存到新的 CSV 文件
final_df.to_csv('processed_val_labels.csv', index=False)
