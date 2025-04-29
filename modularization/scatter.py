import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('resampled_data_hourly.csv')

# 给定最大值（归一化标准）
max_values = {'FW': 2800, 'Feeder': 400, 'W_G': 1000}

# 对数据进行归一化处理
df['FW'] = (df['FW']+0.001) / max_values['FW']
df['Feeder'] = (df['Feeder']+0.001) / max_values['Feeder']
df['W_G'] = (df['W_G']+0.001) / max_values['W_G']

# 给定的示例数据（燃煤量, 有功功率, 给水流量）
# 使用归一化后的最大值来拟合直线
x = np.array([400, 300, 200, 160]) / max_values['Feeder']
y = np.array([1000, 750, 500, 400]) / max_values['W_G']
z = np.array([2800, 1950, 1300, 1040]) / max_values['FW']

# 使用线性回归拟合直线
model = LinearRegression()
model.fit(np.vstack([x, y]).T, z)

# 使用拟合的直线进行预测
z_fit = model.predict(np.vstack([x, y]).T)

# 创建 Plotly 3D 图
fig = go.Figure()

# 绘制原始数据点
fig.add_trace(go.Scatter3d(
    x=df['Feeder'], y=df['W_G'], z=df['FW'],
    mode='markers',
    marker=dict(size=3, color='red'),
    name='数据点'
))

# 绘制拟合的直线
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z_fit,
    mode='lines',
    line=dict(color='blue', width=15),
    name='拟合直线'
))

# 设置图表布局和坐标轴标签
fig.update_layout(
    title="3D 线性拟合直线",
    scene=dict(
        xaxis_title="Feeder ",
        yaxis_title="W_G ",
        zaxis_title="FW"
    ),
    plot_bgcolor="white",  # 设置白色背景
    paper_bgcolor="white",  # 设置纸张背景为白色
    font=dict(family="Microsoft YaHei", color="black", size=15),  # 设置中文字体
    showlegend=True,
)

# 修改保存路径
output_file_path = '3d_normalized_line_fit.html'

# 保存图表为 HTML 文件
fig.write_html(output_file_path)

# 提示用户图表已保存
print(f"图表已保存为 '{output_file_path}'，请打开文件查看互动式图表。")
