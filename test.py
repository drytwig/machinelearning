import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import copy

torch.manual_seed(42)
np.random.seed(42)

class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess(train_path, test_path, is_short_term=True):
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['Date'] = df['DateTime'].dt.date
    df = df.dropna(subset=['Date'])
    
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    
    # 按天聚合
    agg_dict = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }
    daily_df = df.groupby('Date').agg(agg_dict).reset_index().sort_values('Date')
    
    # 计算剩余分表能耗
    daily_df['Sub_metering_remainder'] = (daily_df['Global_active_power'] * 1000 / 60) - \
                                        (daily_df['Sub_metering_1'] + daily_df['Sub_metering_2'] + daily_df['Sub_metering_3'])
    
    # 提取特征和目标变量
    features = daily_df.drop(['Date', 'Global_active_power'], axis=1).values
    target = daily_df['Global_active_power'].values.reshape(-1, 1)
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target).flatten()
    
    # 生成序列
    seq_len = 90
    pred_len = 90 if is_short_term else 365
    X, y = [], []
    max_idx = len(daily_df) - seq_len - pred_len + 1
    for i in range(max_idx):
        X.append(features_scaled[i:i+seq_len])  # 特征序列
        y.append(target_scaled[i+seq_len:i+seq_len+pred_len])  # 目标序列
    
    # 时间序列分割
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    train_dataset = PowerDataset(X_train, y_train)
    test_dataset = PowerDataset(X_test, y_test)
    return train_dataset, test_dataset, scaler_X, scaler_y, pred_len


class CrossAttentionHybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, pred_len=90, use_lstm=True):
        super().__init__()
        self.pred_len = pred_len
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.seq_encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.3,
                bidirectional=False
            )
        else:
            self.seq_encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.3
            )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.2
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        
        # 输出层
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, pred_len)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 时序编码
        if self.use_lstm:
            seq_out, _ = self.seq_encoder(x)  # (batch, seq_len, hidden_dim)
        else:
            seq_out, _ = self.seq_encoder(x)
        seq_out = self.norm1(seq_out)
        
        # 交叉注意力
        query = seq_out
        key = self.key_proj(x)
        value = self.value_proj(x)
        attn_out, _ = self.cross_attn(query, key, value)
        attn_out = self.norm2(attn_out + seq_out)  # 残差连接
        attn_out = self.dropout(attn_out)
        
        # 输出预测
        x = self.relu(self.fc1(attn_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = x[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc3(x)  # (batch, pred_len)
        return out


def train_model(model, train_loader, test_loader, pred_len, scaler_y, epochs=200, lr=0.001, device='cuda'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    best_test_loss = float('inf')
    best_model = None
    patience = 15
    counter = 0 
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        
        # 验证
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                test_loss += loss.item() * X.size(0)
        
        train_loss /= len(train_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = copy.deepcopy(model.state_dict())  # 深拷贝模型参数
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"早停于第{epoch+1}轮（连续{patience}轮测试损失未下降）")
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    model.eval()
    
    # 生成最终预测
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X).cpu().numpy()
            # 反归一化目标值
            y_true.append(scaler_y.inverse_transform(y.numpy().reshape(-1, pred_len)).flatten())
            y_pred.append(scaler_y.inverse_transform(outputs.reshape(-1, pred_len)).flatten())
    
    # 计算评估指标
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return model, mse, mae, y_true, y_pred


if __name__ == "__main__":
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 短期预测
    print("\n===== 短期预测任务 =====")
    train_dataset, test_dataset, scaler_X, scaler_y, pred_len_short = load_and_preprocess(
        "./train.csv", "./test.csv", is_short_term=True
    )
    input_dim = train_dataset[0][0].shape[-1]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    short_mse_list, short_mae_list = [], []
    for run in range(5):
        print(f"\n----- 短期预测 第{run+1}/5轮 -----")
        model = CrossAttentionHybridModel(
            input_dim=input_dim, 
            hidden_dim=256,
            pred_len=pred_len_short, 
            use_lstm=True
        )
        trained_model, mse, mae, _, _ = train_model(
            model, train_loader, test_loader, pred_len_short, scaler_y, 
            epochs=200, lr=0.001, device=device
        )
        short_mse_list.append(mse)
        short_mae_list.append(mae)
        torch.save(trained_model.state_dict(), f"short_term_model_run{run+1}.pth")
    
    # 短期结果统计
    print("\n短期预测结果(5轮平均):")
    print(f"MSE: {np.mean(short_mse_list):.2f} ± {np.std(short_mse_list):.2f}")
    print(f"MAE: {np.mean(short_mae_list):.2f} ± {np.std(short_mae_list):.2f}")
    

    # 长期预测
    print("\n===== 长期预测任务 =====")
    train_dataset_long, test_dataset_long, _, scaler_y_long, pred_len_long = load_and_preprocess(
        "train.csv", "test.csv", is_short_term=False
    )
    train_loader_long = DataLoader(train_dataset_long, batch_size=batch_size, shuffle=True)
    test_loader_long = DataLoader(test_dataset_long, batch_size=batch_size, shuffle=False)
    
    long_mse_list, long_mae_list = [], []
    for run in range(5):
        print(f"\n----- 长期预测 第{run+1}/5轮 -----")
        model = CrossAttentionHybridModel(
            input_dim=input_dim, 
            hidden_dim=256, 
            pred_len=pred_len_long, 
            use_lstm=True
        )
        trained_model, mse, mae, _, _ = train_model(
            model, train_loader_long, test_loader_long, pred_len_long, scaler_y_long,
            epochs=200, lr=0.001, device=device
        )
        long_mse_list.append(mse)
        long_mae_list.append(mae)
        torch.save(trained_model.state_dict(), f"long_term_model_run{run+1}.pth")
    
    # 长期结果统计
    print("\n长期预测结果(5轮平均):")
    print(f"MSE: {np.mean(long_mse_list):.2f} ± {np.std(long_mse_list):.2f}")
    print(f"MAE: {np.mean(long_mae_list):.2f} ± {np.std(long_mae_list):.2f}")
    
    print("\n绘制短期预测对比图...")
    model_short = CrossAttentionHybridModel(input_dim=input_dim, pred_len=pred_len_short)
    model_short.load_state_dict(torch.load("short_term_model_run1.pth", map_location=device))
    model_short.to(device)
    model_short.eval()
    
    # 获取测试集样本并预测
    X_test_sample, y_test_sample = next(iter(test_loader))
    with torch.no_grad():
        y_pred_sample = model_short(X_test_sample.to(device)).cpu().numpy()
    
    # 反归一化样本预测值
    y_test_sample_original = scaler_y.inverse_transform(y_test_sample.numpy()[0].reshape(1, -1)).flatten()
    y_pred_sample_original = scaler_y.inverse_transform(y_pred_sample[0].reshape(1, -1)).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_sample_original, label='真实值', linewidth=2)
    plt.plot(y_pred_sample_original, label='预测值', linestyle='--', linewidth=2)
    plt.title('短期预测曲线对比(90)', fontsize=14)
    plt.xlabel('天数', fontsize=12)
    plt.ylabel('Global_active_power(原始尺度)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('pytorch_short_term_pred.png', dpi=300)
    print("短期预测曲线已保存为 'pytorch_short_term_pred.png'")
    
    print("\n绘制长期预测对比图...")
    model_long = CrossAttentionHybridModel(input_dim=input_dim, pred_len=pred_len_long)
    model_long.load_state_dict(torch.load("long_term_model_run1.pth", map_location=device))
    model_long.to(device)
    model_long.eval()
    
    # 获取长期测试集样本
    X_test_long_sample, y_test_long_sample = next(iter(test_loader_long))
    with torch.no_grad():
        y_pred_long_sample = model_long(X_test_long_sample.to(device)).cpu().numpy()
    
    y_test_long_original = scaler_y_long.inverse_transform(y_test_long_sample.numpy()[0].reshape(1, -1)).flatten()
    y_pred_long_original = scaler_y_long.inverse_transform(y_pred_long_sample[0].reshape(1, -1)).flatten()
    
    plt.figure(figsize=(15, 6)) 
    plt.plot(y_test_long_original, label='真实值', linewidth=2)
    plt.plot(y_pred_long_original, label='预测值', linestyle='--', linewidth=2)
    plt.title('长期预测曲线对比(365天)', fontsize=14)
    plt.xlabel('天数', fontsize=12)
    plt.ylabel('Global_active_power(原始尺度)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('pytorch_long_term_pred.png', dpi=300)
    print("长期预测曲线已保存为 'pytorch_long_term_pred.png'")