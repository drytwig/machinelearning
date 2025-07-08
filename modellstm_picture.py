import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# === 1. 加载并处理数据 ===
df = pd.read_csv("F:\PycharmProjects\kerass\deal_data\daily_summary.csv")
df['DateTime'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['DateTime'])
df['Date'] = df['DateTime'].dt.date
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
daily = df.groupby('Date')['Global_active_power'].sum().reset_index()
daily = daily.set_index('Date')

# === 2. 数据归一化 ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily)

# === 3. 构造序列数据集 ===
def create_dataset(data, lookback, predict_forward):
    X, y = [], []
    for i in range(len(data) - lookback - predict_forward):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+predict_forward])
    return np.array(X), np.array(y)

# 参数设置
LOOKBACK = 90
PREDICT_FORWARD_SHORT = 90
PREDICT_FORWARD_LONG = 365

# 短期预测数据准备
X_short, y_short = create_dataset(scaled_data, LOOKBACK, PREDICT_FORWARD_SHORT)

# 长期预测数据准备
X_long, y_long = create_dataset(scaled_data, LOOKBACK, PREDICT_FORWARD_LONG)

# === 4. 构建 LSTM 模型函数 ===
def build_lstm_model(input_shape, output_dim):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 5. 短期预测模型训练 ===
model_short = build_lstm_model((LOOKBACK, 1), PREDICT_FORWARD_SHORT)
model_short.fit(X_short, y_short, epochs=20, batch_size=16)

# === 6. 长期预测模型训练 ===
model_long = build_lstm_model((LOOKBACK, 1), PREDICT_FORWARD_LONG)
model_long.fit(X_long, y_long, epochs=20, batch_size=16)

# === 7. 做预测（从最后一个窗口开始预测）===
last_input = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
short_prediction = model_short.predict(last_input)
long_prediction = model_long.predict(last_input)

# 反归一化
short_pred = scaler.inverse_transform(short_prediction.reshape(-1, 1))
long_pred = scaler.inverse_transform(long_prediction.reshape(-1, 1))

# === 8. 结果可视化 ===
plt.figure(figsize=(14, 6))
plt.plot(range(len(short_pred)), short_pred, label='90-day Prediction')
plt.title("Short-Term Power Forecast (90 days)")
plt.xlabel("Day")
plt.ylabel("Global Active Power")
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(range(len(long_pred)), long_pred, label='365-day Prediction', color='orange')
plt.title("Long-Term Power Forecast (365 days)")
plt.xlabel("Day")
plt.ylabel("Global Active Power")
plt.legend()
plt.show()
