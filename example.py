import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


# Step 1: Load JSON and save it as CSV for each JSON file
def json_to_csv(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract relevant information (assuming the structure provided earlier)
    tagvals = data['tagvals']

    # Convert to DataFrame
    df = pd.DataFrame(tagvals)

    # Save to CSV
    csv_file_path = json_file_path.replace('.json', '.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved to: {csv_file_path}")
    return csv_file_path


# Step 2: Merge CSV files (assuming the structure provided)
def merge_csv_files():
    df1 = pd.read_csv('LW.TT.N6TS_F_FW.csv')
    df2 = pd.read_csv('LW.BT.N6BC_F_Feeder.csv')
    df3 = pd.read_csv('LW.ET.N6ES_W_G1.csv')

    df1 = df1[['time', 'val']].rename(columns={'val': 'FW'})
    df2 = df2[['time', 'val']].rename(columns={'val': 'Feeder'})
    df3 = df3[['time', 'val']].rename(columns={'val': 'W_G'})

    df1['time'] = pd.to_datetime(df1['time'], unit='ms')
    df2['time'] = pd.to_datetime(df2['time'], unit='ms')
    df3['time'] = pd.to_datetime(df3['time'], unit='ms')

    merged_df = pd.merge(df1, df2, on='time', how='outer')
    merged_df = pd.merge(merged_df, df3, on='time', how='outer')

    return merged_df


# Step 3: Resample the data
def resample_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df_resampled = df.resample('h', on='time').mean()
    df_resampled = df_resampled.dropna()
    return df_resampled


# Step 4: Normalize the data
def normalize_data(df):
    max_values = {'FW': 2800, 'Feeder': 400, 'W_G': 1000}

    df['FW'] = (df['FW'] + 0.0001) / max_values['FW']
    df['Feeder'] = (df['Feeder'] + 0.0001) / max_values['Feeder']
    df['W_G'] = (df['W_G'] + 0.0001) / max_values['W_G']

    return df


# Step 5: Fit the linear regression model and compute the distance
def fit_linear_model(df):
    x = np.array([400, 300, 200, 160]) / 400
    y = np.array([1000, 750, 500, 400]) / 1000
    z = np.array([2800, 1950, 1300, 1040]) / 2800

    model = LinearRegression()
    model.fit(np.vstack([x, y]).T, z)

    a, b = model.coef_
    c = model.intercept_

    def compute_distance(xi, yi, zi, a, b, c):
        return abs(a * xi + b * yi + c - zi) / np.sqrt(a ** 2 + b ** 2 + 1)

    distances = df.apply(lambda row: compute_distance(row['Feeder'], row['W_G'], row['FW'], a, b, c), axis=1)
    df['Distance_to_Line'] = distances

    # Compute the average distance error
    avg_distance = df['Distance_to_Line'].mean()
    print(f"平均误差距离: {avg_distance}")

    return df


# Step 6: Create the 3D plot
def create_3d_plot(df):
    max_values = {'FW': 2800, 'Feeder': 400, 'W_G': 1000}

    x = np.array([400, 300, 200, 160]) / max_values['Feeder']
    y = np.array([1000, 750, 500, 400]) / max_values['W_G']
    z = np.array([2800, 1950, 1300, 1040]) / max_values['FW']

    model = LinearRegression()
    model.fit(np.vstack([x, y]).T, z)
    z_fit = model.predict(np.vstack([x, y]).T)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df['Feeder'], y=df['W_G'], z=df['FW'],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='数据点'
    ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z_fit,
        mode='lines',
        line=dict(color='blue', width=15),
        name='拟合直线'
    ))

    fig.update_layout(
        title="3D 线性拟合直线",
        scene=dict(
            xaxis_title="Feeder ",
            yaxis_title="W_G ",
            zaxis_title="FW"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Microsoft YaHei", color="black", size=15),
        showlegend=True,
    )

    fig.write_html('FW_W_G1.html')
    print("图表已保存为 'FW_W_G1.html'，请打开文件查看互动式图表。")


# Main execution sequence
def main():
    # Convert JSON files to CSV first
    json_files = [
        'LW.BT.N6BC_F_Feeder.json',
        'LW.ET.N6ES_W_G1.json',
        'LW.TT.N6TS_F_FW.json'
    ]

    for json_file in json_files:
        json_to_csv(json_file)

    # Then proceed with the CSV processing
    merged_df = merge_csv_files()
    resampled_df = resample_data(merged_df)
    normalized_df = normalize_data(resampled_df)
    distance_df = fit_linear_model(normalized_df)

    # Save only the final distance data to CSV
    distance_df.to_csv('FW_W_G1.csv', index=False)
    print("距离数据已保存为 'FW_W_G1.csv'")

    # Create the 3D plot
    create_3d_plot(resampled_df)


if __name__ == '__main__':
    main()
