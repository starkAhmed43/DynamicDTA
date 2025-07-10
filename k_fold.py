import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np
import os
# 设置随机种子
np.random.seed(42)

# 读取并打乱数据集
df = pd.read_csv('IC50.csv')
df = shuffle(df, random_state=42)

# 手动实现归一化
for column in ['Avg.RMSF', 'Avg.gyr', 'Div.SE', 'Div.MM']:
    min_value = df[column].min()
    max_value = df[column].max()
    df[column] = (df[column] - min_value) / (max_value - min_value)

# 创建5折交叉验证的对象
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 使用归一化后的数据进行5折划分，并保存到对应目录
for fold, (train_index, test_index) in enumerate(kf.split(df)):
    # 获取当前折的训练集和测试集
    train_data, test_data = df.iloc[train_index], df.iloc[test_index]

    # 创建对应目录（如果目录不存在，则创建）
    fold_dir = f'fold{fold + 1}'
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # 将训练集和测试集分别保存到对应的目录下
    train_data.to_csv(os.path.join(fold_dir, 'data_train.csv'), index=False)
    test_data.to_csv(os.path.join(fold_dir, 'data_test.csv'), index=False)

print("5折交叉验证的数据集已生成并保存在对应的目录中。")
