import pandas as pd

# 读取数据
df = pd.read_csv("F:\PycharmProjects\kerass\deal_data\\train.csv")  # 替换为你的文件名

# 解析时间
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.dropna(subset=['DateTime'])

# 提取日期
df['Date'] = df['DateTime'].dt.date

# 强制转换为 float 类型的列
columns_to_numeric = [
    'Global_active_power', 'Global_reactive_power',
    'Sub_metering_1', 'Sub_metering_2',
    'Voltage', 'Global_intensity',
    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]

for col in columns_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # 将非数字转换为 NaN

# 再做按天聚合
daily_summary = df.groupby('Date').agg({
    'Global_active_power': 'sum',
    'Global_reactive_power': 'sum',
    'Sub_metering_1': 'sum',
    'Sub_metering_2': 'sum',
    'Voltage': 'mean',
    'Global_intensity': 'mean',
    'RR': 'first',
    'NBJRR1': 'first',
    'NBJRR5': 'first',
    'NBJRR10': 'first',
    'NBJBROU': 'first'
}).reset_index()

# 保存或打印结果
daily_summary.to_csv("daily_summary.csv", index=False)
print(daily_summary.head())