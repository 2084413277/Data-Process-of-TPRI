import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('resampled_data_hourly.csv')

# 给定最大值（归一化标准）
max_values = {'FW': 2800, 'Feeder': 400, 'W_G': 1000}

# 对数据进行归一化处理
df['FW'] = (df['FW'] + 0.0001) / max_values['FW']
df['Feeder'] = (df['Feeder'] + 0.0001) / max_values['Feeder']
df['W_G'] = (df['W_G'] + 0.0001) / max_values['W_G']

# 给定的示例数据（燃煤量, 有功功率, 给水流量）
# 使用归一化后的最大值来拟合直线
x = np.array([400, 300, 200, 160]) / max_values['Feeder']
y = np.array([1000, 750, 500, 400]) / max_values['W_G']
z = np.array([2800, 1950, 1300, 1040]) / max_values['FW']

# 使用线性回归拟合直线
model = LinearRegression()
model.fit(np.vstack([x, y]).T, z)

# 获取直线的系数
a, b = model.coef_
c = model.intercept_

# 使用拟合的直线进行预测
z_fit = model.predict(np.vstack([x, y]).T)

# 计算每个点到拟合直线的最短距离
def compute_distance(xi, yi, zi, a, b, c):
    return abs(a * xi + b * yi + c - zi) / np.sqrt(a**2 + b**2 + 1)

# 计算每个点到拟合直线的距离
distances = df.apply(lambda row: compute_distance(row['Feeder'], row['W_G'], row['FW'], a, b, c), axis=1)

# 将距离添加到数据框
df['Distance_to_Line'] = distances

# 保存结果到 CSV 文件
output_file_path = 'distance_to_line.csv'
df.to_csv(output_file_path, index=False)

# 提示用户图表和数据已保存
print(f"距离数据已保存为 '{output_file_path}'。")
