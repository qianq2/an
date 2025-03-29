import torch
from collections import Counter


class TextPreprocessor:
    def __init__(self, file_path, seq_length=20):
        self.seq_length = seq_length
        # 读取文本并统一为小写（减少词汇量）
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        # 构建字符到索引的映射（如{'a':0, 'b':1}）
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def create_sequences(self):
        # 将文本转换为数字序列（如"hello" → [2, 3, 4, 4, 5]）
        encoded_text = [self.char_to_idx[c] for c in self.text]
        sequences = []
        targets = []
        # 滑动窗口生成输入-目标对（输入前20字符，预测第21字符）
        for i in range(len(encoded_text) - self.seq_length):
            seq = encoded_text[i:i + self.seq_length]
            target = encoded_text[i + self.seq_length]
            sequences.append(seq)
            targets.append(target)
        return torch.tensor(sequences), torch.tensor(targets)


if __name__ == "__main__":
    preprocessor = TextPreprocessor("data/西游记.txt")
    sequences, targets = preprocessor.create_sequences()
    print("字符表大小:", len(preprocessor.chars))
    print("第一个序列:", sequences[0])