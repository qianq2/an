import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        """
                中文分字LSTM模型
                参数：
                    vocab_size: 词表大小（来自TextPreprocessor）
                    embedding_dim: 字嵌入维度（建议256）
                    hidden_dim: LSTM隐藏层维度（建议512）
                    num_layers: LSTM层数（默认2层）
                """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
                            input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0
                            )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # 输入x形状：(batch_size, seq_length)
        x = self.embedding(x)  # 输出形状：(batch_size, seq_length, embedding_dim)
        # LSTM处理序列，输出形状：(batch_size, seq_length, hidden_dim)
        out, hidden = self.lstm(x, hidden)
        # 仅取最后一个时间步的输出（out[:, -1, :]）
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # 输出形状：(batch_size, vocab_size)
        return out, hidden


#if __name__ == "__main__":
#   model = CharLSTM(vocab_size=50)
#    print(model)