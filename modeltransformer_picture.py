import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer 模型
class TransformerForecast(nn.Module):
    def __init__(self, input_len, output_len, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, output_len)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.decoder(x)

# 数据加载与处理
def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df['DateTime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=['DateTime'])
    df['Date'] = df['DateTime'].dt.date
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    daily = df.groupby('Date')['Global_active_power'].sum().reset_index()
    return daily.set_index('Date')

# 构造滑动窗口序列
def create_sequences(series, input_len, output_len):
    X, y = [], []
    for i in range(len(series) - input_len - output_len):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

# 训练与评估函数（含预测曲线图）
def train_and_evaluate(data, input_len, output_len, repeat=5, epochs=30, lr=1e-3):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    X, y = create_sequences(scaled, input_len, output_len)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    mse_list, mae_list = [], []
    best_preds = None
    best_targets = None

    for i in range(repeat):
        model = TransformerForecast(input_len, output_len).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_test)

        pred_inv = scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()
        y_inv = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_inv, pred_inv)
        mae = mean_absolute_error(y_inv, pred_inv)

        if i == 0:
            best_preds = pred_inv
            best_targets = y_inv

        mse_list.append(mse)
        mae_list.append(mae)
        print(f"[Round {i+1}] MSE={mse:.2f}, MAE={mae:.2f}")

    print(f"\n=== 总结 ({output_len}天预测) ===")
    print(f"MSE Mean: {np.mean(mse_list):.2f}, Std: {np.std(mse_list):.2f}")
    print(f"MAE Mean: {np.mean(mae_list):.2f}, Std: {np.std(mae_list):.2f}")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(best_targets[:output_len], label='True')
    plt.plot(best_preds[:output_len], label='Predicted')
    plt.title(f"Transformer Forecast ({output_len} days)")
    plt.xlabel("Day")
    plt.ylabel("Global Active Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主程序
def main():
    file_path = "/public/home/zmh/ml/deal_data/train.csv"
    daily_data = load_and_prepare(file_path)

    print("\n🔹 短期预测（90天）")
    train_and_evaluate(daily_data, input_len=90, output_len=90)

    print("\n🔸 长期预测（365天）")
    train_and_evaluate(daily_data, input_len=90, output_len=365)

if __name__ == "__main__":
    main()
