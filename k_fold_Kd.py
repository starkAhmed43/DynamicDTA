import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

# 设置随机种子
np.random.seed(42)

# 读取并打乱数据集
df = pd.read_csv('kd.csv')
df = shuffle(df, random_state=42)

# 创建5折交叉验证的对象
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 使用归一化后的数据进行5折划分，并保存到当前路径
for fold, (train_index, test_index) in enumerate(kf.split(df)):
    # 获取当前折的训练集和测试集
    train_data, test_data = df.iloc[train_index], df.iloc[test_index]

    # 将训练集和测试集分别保存到当前路径下
    train_data.to_csv(f'train_data_fold_{fold + 1}.csv', index=False)
    test_data.to_csv(f'test_data_fold_{fold + 1}.csv', index=False)

print("5折交叉验证的数据集已生成并保存在当前路径中。")

