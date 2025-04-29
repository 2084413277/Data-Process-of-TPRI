import pandas as pd

# 读取三个CSV文件
df1 = pd.read_csv('LW.TT.N6TS_F_FW.csv')
df2 = pd.read_csv('LW.BT.N6BC_F_Feeder.csv')
df3 = pd.read_csv('LW.ET.N6ES_W_G.csv')

# 提取time和val列，并转换时间格式
df1 = df1[['time', 'val']].rename(columns={'val': 'FW'})
df2 = df2[['time', 'val']].rename(columns={'val': 'Feeder'})
df3 = df3[['time', 'val']].rename(columns={'val': 'W_G'})

# 将时间戳转换为常规时间格式
df1['time'] = pd.to_datetime(df1['time'], unit='ms')
df2['time'] = pd.to_datetime(df2['time'], unit='ms')
df3['time'] = pd.to_datetime(df3['time'], unit='ms')

# 合并三个DataFrame，按time列合并
merged_df = pd.merge(df1, df2, on='time', how='outer')
merged_df = pd.merge(merged_df, df3, on='time', how='outer')

# 保存合并后的数据
merged_df.to_csv('merged_data.csv', index=False)

print("合并完成，结果已保存为 'merged_data.csv'")
