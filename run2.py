import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# 计算标准差
def calculate_std(actuals, predictions):
    return np.std(actuals - predictions)


# ========== 定义 Transformer 模型 ==========
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=8):
        super(TransformerModel, self).__init__()

        # Transformer的输入需要进行Embedding
        self.embedding = nn.Linear(input_size, hidden_size)

        # Transformer Encoder 层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers,
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 将输入数据从 (batch_size, seq_len, input_size) 转换为 (seq_len, batch_size, input_size)
        x = x.permute(1, 0, 2)

        # 通过Embedding层
        x = self.embedding(x)

        # 通过Transformer Encoder
        x = self.transformer(x)

        # 取最后一个时间步的输出
        out = self.fc(x[-1, :, :])  # 仅使用最后一个时间步的输出

        return out


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main(
    n_hours=1, num_rounds=5, hidden_size=64, num_layers=1, num_heads=8
):  # 添加了 hidden_size、num_layers 和 num_heads 参数
    # ========== 数据预处理 ==========
    # 加载数据
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")

    # 如果有日期列，例如 'date' 列，将其解析为时间戳
    if "date" in train_data.columns:  # 替换 'date' 为实际日期列名
        train_data["date"] = pd.to_datetime(train_data["date"])
        test_data["date"] = pd.to_datetime(test_data["date"])

        # 将日期转为时间戳
        train_data["date"] = train_data["date"].map(pd.Timestamp.timestamp)
        test_data["date"] = test_data["date"].map(pd.Timestamp.timestamp)

    # 删除非数值列或对其进行编码
    non_numeric_columns = train_data.select_dtypes(exclude=["number"]).columns
    print("非数值列：", non_numeric_columns)

    for col in non_numeric_columns:
        encoder = LabelEncoder()

        # 合并训练和测试数据，确保编码一致
        combined_data = pd.concat([train_data[col], test_data[col]], axis=0)
        encoder.fit(combined_data)

        # 分别对训练和测试数据进行编码
        train_data[col] = encoder.transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])

    # 替换无效值并删除缺失值
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # 分离特征和标签
    X_train = train_data.iloc[:, :-1].values  # 假设最后一列是标签
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ========== 滑动窗口数据集 ==========
    # 创建滑动窗口的数据集
    def create_sliding_window_data(X, y, window_size, forecast_size):
        X_windowed, y_windowed = [], []
        for i in range(window_size, len(X) - forecast_size + 1):
            X_windowed.append(X[i - window_size : i])  # 过去 window_size 小时的特征
            y_windowed.append(
                y[i : i + forecast_size]
            )  # 预测接下来的 forecast_size 小时的数据
        return np.array(X_windowed), np.array(y_windowed)

    window_size = 96  # 过去96小时的数据
    X_train_windowed, y_train_windowed = create_sliding_window_data(
        X_train, y_train, window_size, n_hours
    )
    X_test_windowed, y_test_windowed = create_sliding_window_data(
        X_test, y_test, window_size, n_hours
    )

    # ========== 数据集和数据加载器 ==========
    # 创建数据集
    train_dataset = CustomDataset(X_train_windowed, y_train_windowed)
    test_dataset = CustomDataset(X_test_windowed, y_test_windowed)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 实例化 Transformer 模型
    input_size = X_train_windowed.shape[2]  # 每个时间步的特征数量
    model = TransformerModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=n_hours,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # ========== 定义损失函数和优化器 ==========
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ========== 多轮实验 ==========
    all_predictions = []
    all_actuals = []

    for round_num in range(num_rounds):
        print(f"实验轮数 {round_num + 1}/{num_rounds}")

        # 重新初始化模型和优化器
        model = TransformerModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=n_hours,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                # Transformer 需要输入形状为 (batch_size, seq_len, input_size)
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # 调整输入形状
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
            )

        # 测试模型
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # 调整输入形状
                outputs = model(inputs)
                predictions.append(outputs.numpy())
                actuals.append(targets.numpy())

        # 将预测值和实际值转换为一维数组
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        # 计算均方误差和平均绝对误差
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        print(f"测试集 MSE: {mse:.4f}, MAE: {mae:.4f}")

        # 在最后一轮实验后记录预测结果
        if round_num == num_rounds - 1:
            all_predictions = predictions
            all_actuals = actuals

    # ========== 绘制曲线图 ==========
    plt.figure(figsize=(10, 6))

    # 只取前30个点
    num_points = 96

    # 假设绘制最后一轮实验的预测与实际值对比
    plt.plot(all_actuals[:num_points, 0], label="actual", color="blue", linewidth=2)
    plt.plot(
        all_predictions[:num_points, 0],
        label="predicted",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    plt.legend()
    plt.xlabel("instant")
    plt.ylabel("cnt")
    plt.grid()
    plt.show()
    print(
        "Standard Deviation:",
        calculate_std(all_actuals[:num_points, 0], all_predictions[:num_points, 0]),
    )


if __name__ == "__main__":
    main(n_hours=240, num_rounds=5)  # 设置进行五轮实验
