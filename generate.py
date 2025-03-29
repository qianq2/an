from model.model import CharLSTM
from utils import TextPreprocessor
from config import *
import torch
import random

def generate_text(model, start_str, length=100, temperature=0.8):
    model.eval()  # 切换为评估模式（关闭Dropout等）
    chars = [c for c in start_str.lower()]
    hidden = None  # 初始隐藏状态为None
    for _ in range(length):
        # 取最后seq_length个字符作为输入
        seq = chars[-SEQ_LENGTH:]
        # 将字符转换为索引张量（形状：[1, seq_length]）
        seq_tensor = torch.tensor([preprocessor.char_to_idx[c] for c in seq]).unsqueeze(0)
        # 预测下一个字符的概率分布
        with torch.no_grad():  # 禁用梯度计算，节省内存
            output, hidden = model(seq_tensor, hidden)
        # 通过温度参数调整概率分布的平滑度
        probs = torch.softmax(output / temperature, dim=1)
        # 从概率分布中采样一个字符索引
        next_char_idx = torch.multinomial(probs, 1).item()
        chars.append(preprocessor.idx_to_char[next_char_idx])
    return ''.join(chars)

# 加载模型和预处理配置
preprocessor = TextPreprocessor("data/西游记.txt")
model = CharLSTM(vocab_size=len(preprocessor.chars))
model.load_state_dict(torch.load("model/LSTM_model.pth"))

# 示例生成
print(generate_text(model, start_str="猴哥", temperature=0.5))