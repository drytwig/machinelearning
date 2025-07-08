import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ====================== 数据准备 =======================
def load_data(path):
    df = pd.read_csv(path)
    df['DateTime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=['DateTime'])
    df['Date'] = df['DateTime'].dt.date
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    daily = df.groupby('Date')['Global_active_power'].sum().reset_index()
    daily = daily.set_index('Date')
    return daily

def create_dataset(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+forecast_horizon])
    return np.array(X), np.array(y)

# ====================== 模型构建 =======================
def build_model(input_shape, output_dim):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ====================== 实验主程序 =======================
def run_experiment(data, lookback, forecast_horizon, repeat=5, epochs=20, batch_size=16):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_dataset(scaled_data, lookback, forecast_horizon)

    # 划分训练集和测试集（80%训练，20%测试）
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    mse_list, mae_list = [], []

    for i in range(repeat):
        model = build_model((lookback, 1), forecast_horizon)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        pred = model.predict(X_test)
        # 保证维度为 2D (samples × forecast_horizon)
        pred_flat = pred.reshape(-1, 1)
        y_test_flat = y_test.reshape(-1, 1)

        pred_rescaled = scaler.inverse_transform(pred_flat)
        y_test_rescaled = scaler.inverse_transform(y_test_flat)

        mse = mean_squared_error(y_test_rescaled, pred_rescaled)
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)

        mse_list.append(mse)
        mae_list.append(mae)
        print(f"Round {i+1}: MSE={mse:.4f}, MAE={mae:.4f}")

    print("\n=== 5轮实验结果统计 ===")
    print(f"MSE: Mean={np.mean(mse_list):.4f}, Std={np.std(mse_list):.4f}")
    print(f"MAE: Mean={np.mean(mae_list):.4f}, Std={np.std(mae_list):.4f}")

# ====================== 执行 =======================
data = load_data("F:\PycharmProjects\kerass\deal_data\daily_summary.csv")

# 短期预测（90天）
print(">>> 短期预测（90天）")
run_experiment(data, lookback=90, forecast_horizon=90)

# 长期预测（365天）
print("\n>>> 长期预测（365天）")
run_experiment(data, lookback=90, forecast_horizon=365)
