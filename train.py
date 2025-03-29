from model.model import CharLSTM
from utils import TextPreprocessor
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import gradio as gr
from torch.utils.data import DataLoader, TensorDataset


# 数据加载与预处理
preprocessor = TextPreprocessor("data/西游记.txt")
sequences, targets = preprocessor.create_sequences()
# 创建数据集和数据加载器（自动分批次）
dataset = TensorDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 模型初始化
model = CharLSTM(vocab_size=len(preprocessor.chars))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()  # 分类任务常用损失函数

# 训练循环
for epoch in range(EPOCHS):
    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()  # 清空梯度
        output, _ = model(x)   # 前向传播
        loss = criterion(output, y)  # 计算损失
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新权重
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x, y = x.to(device), y.to(device)

# 保存模型
torch.save(model.state_dict(), "model/LSTM_model.pth")