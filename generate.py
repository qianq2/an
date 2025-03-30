
from model.model import CharLSTM
from utils import TextPreprocessor
from config import *
import torch


def generate_text(model, preprocessor, start_str, length=100, temperature=0.8):
    """
    生成文本
    参数：
        model: 训练好的模型
        preprocessor: 数据预处理器
        start_str: 起始字符串
        length: 生成长度
        temperature: 温度参数（>1更随机，<1更保守）
    """
    model.eval()  # 切换为评估模式（关闭Dropout等）
    device = next(model.parameters()).device
    # 处理起始字符串
    chars = list(start_str)
    if len(chars) < SEQ_LENGTH:
        chars = ['<PAD>'] * (SEQ_LENGTH - len(chars)) + chars
    else:
        chars = chars[-SEQ_LENGTH]
    hidden = None  # 初始隐藏状态为None

    for _ in range(length):
        # 编码当前序列
        encoded = preprocessor.encode(chars)
        x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

        # 预测下一个字符
        with torch.no_grad():  # 禁用梯度计算，节省内存
            output, hidden = model(x, hidden)
        # 应用温度采样
        probs = torch.softmax(output / temperature, dim=1).squeeze()
        next_char_idx = torch.multinomial(probs, 1).item()
        chars.append(preprocessor.idx2char[next_char_idx])
    # 解码并去除填充符
    generated = ''.join(chars).replace('<PAD>', '')
    return generated

if __name__ == "__main__":
    # 加载模型和预处理配置
    preprocessor = TextPreprocessor("data/西游记.txt", seq_length=SEQ_LENGTH)
    model = CharLSTM(preprocessor.vocab_size)
    model.load_state_dict(torch.load("model/best_model.pth", map_location='cpu'))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 示例生成
    seed_text = "猴哥"
    generated_text = generate_text(
        model,
        preprocessor,
        start_str=seed_text,
        temperature=0.7,
        length=200
    )
    print("生成结果:\n" + generated_text)