import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        # 嵌入层：将字符索引转换为密集向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列信息，输出隐藏状态
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层：将LSTM输出映射到字符表概率分布
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # 输入x形状：(batch_size, seq_length)
        x = self.embedding(x)  # 输出形状：(batch_size, seq_length, embedding_dim)
        # LSTM处理序列，输出形状：(batch_size, seq_length, hidden_dim)
        out, hidden = self.lstm(x, hidden)
        # 仅取最后一个时间步的输出（out[:, -1, :]）
        out = self.fc(out[:, -1, :])  # 输出形状：(batch_size, vocab_size)
        return out, hidden


if __name__ == "__main__":
    model = CharLSTM(vocab_size=50)
    print(model)