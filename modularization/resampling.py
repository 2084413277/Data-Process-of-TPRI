import pandas as pd

# 读取合并后的CSV文件
df = pd.read_csv('merged_data.csv')

# 转换'time'列为日期时间格式
df['time'] = pd.to_datetime(df['time'])

# 按每小时分组，并计算每列的均值
df_resampled = df.resample('h', on='time').mean()

# 删除包含缺失值的行
df_resampled = df_resampled.dropna()

# 保存新的CSV文件
df_resampled.to_csv('resampled_data_hourly.csv', index=True)

print("按每小时均值汇总并删除缺失值的行，文件已保存为 'resampled_data_hourly.csv'")
