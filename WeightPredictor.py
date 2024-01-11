import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('1.csv')


target_col = 'Daily gas capacity(m3)'
target_data = data[target_col].values.reshape(-1, 1)


scaler = MinMaxScaler()
target_data = scaler.fit_transform(target_data)
class test1(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, output_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        )

    def forward(self, x):
        out = self.tcn(x)
        return out[:, :, -1]
class WeightPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(WeightPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 1  
output_size = 1  
num_channels = 128  
kernel_size = 5 
dropout = 0.2  
hidden_size = 64  
lr = 0.001  
epochs = 100 


target_tensor = torch.Tensor(target_data).unsqueeze(1)


train_size = int(len(target_tensor) * 0.8)
train_data = target_tensor[:train_size]
test_data = target_tensor[train_size:]

tcn_model = test1(input_size, output_size, num_channels, kernel_size, dropout)
weight_predictor = WeightPredictor(input_size, hidden_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(list(tcn_model.parameters()) + list(weight_predictor.parameters()), lr=lr)


for epoch in range(epochs):
    optimizer.zero_grad()

    # 使用TCN模型生成预测值
    tcn_output = tcn_model(train_data)

    # 使用权重预测器模型生成权重
    weights = weight_predictor(train_data)

    # 对预测值进行加权求和
    weighted_output = torch.sum(tcn_output * weights, dim=1)

    # 计算损失
    loss = criterion(weighted_output, train_data[:, 0])

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

tcn_model.eval()
weight_predictor.eval()

tcn_output = tcn_model(test_data)

# 使用权重预测器模型生成权重
weights = weight_predictor(test_data)

# 对预测值进行加权求和
weighted_output = torch.sum(tcn_output * weights, dim=1)


predictions = weighted_output.detach().numpy()

predictions = scaler.inverse_transform(predictions)

print(predictions)

