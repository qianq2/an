import torch
import re
from collections import Counter



class TextPreprocessor:
    def __init__(self, file_path, seq_length=50, min_freq=2):
        """
                中文分字数据预处理类
                参数：
                    file_path: 文本文件路径（如`data/西游记.txt`）
                    seq_length: 输入序列长度（建议50）
                    min_freq: 字符最小出现频次，低于此值视为低频字（过滤为<UNK>）
        """
        self.seq_length = seq_length
        self.min_freq = min_freq
        # 读取并清洗文本
        with open(file_path, 'r', encoding='UTF-8') as f:
            raw_text = f.read()
        self.clean_text = self._clean_chinese_text(raw_text)

        # 分字并构建词表
        self.chars = list(self.clean_text)
        self._build_vocab()


    def _clean_chinese_text(self, text):
        """清洗中文文本：保留汉字、常用标点和数字（替换为#NUM）"""
        # 保留字符：汉字、常见中文标点、数字（按需调整）
        kept_chars = r'\u4e00-\u9fa5，。！？、；：“”‘’（）【】…—～《》\n'
        text = re.sub(f'[^{kept_chars}0-9]', '', text)
        # 统一数字为特殊标记（可选）
        text = re.sub(r'\d+', '#NUM', text)
        return text

    def _build_vocab(self):
        """构建字符词表（过滤低频字）"""
        char_counts = Counter(self.chars)
        # 按频次排序并过滤低频字
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        self.vocab = ['<PAD>', '<UNK>']  # 填充符和未知符
        self.vocab += [char for char, count in sorted_chars if count >= self.min_freq]

        # 构建字符到索引的映射
        self.char2idx = {char : idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx : char for idx, char in enumerate(self.vocab)}
    def encode(self, text):
        """将文本转换为索引列表（未知字替换为<UNK>）"""
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]

    def decode(self, indices):
        """将索引列表转换回文本"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])

    def create_sequences(self):
        """生成训练用的输入-目标序列对"""
        encoded = self.encode(self.clean_text)
        sequences = []
        targets = []

        for i in range(len(encoded) - self.seq_length):
            sequences.append(encoded[i:i + self.seq_length])
            targets.append(encoded[i + self.seq_length])
        return torch.tensor(sequences), torch.tensor(targets)

    @property
    def vocab_size(self):
        return len(self.vocab)

if __name__ == "__main__":
    preprocessor = TextPreprocessor("data/西游记.txt", seq_length=50, min_freq=2)

    # 2. 查看词表信息
    print(f"总字符数：{len(preprocessor.chars)}")
    print(f"词表大小（含<PAD>/<UNK>）：{preprocessor.vocab_size}")
    print("前10个高频字：", preprocessor.vocab[2:12])  # 前两位是<PAD>和<UNK>

    # 3. 生成训练序列
    sequences, targets = preprocessor.create_sequences()
    print("\n输入序列形状：", sequences.shape)  # [num_samples, seq_length]
    print("目标序列形状：", targets.shape)  # [num_samples]

    # 4. 测试编码-解码
    test_text = "江湖风云再起，少年握紧手中的剑。"
    encoded = preprocessor.encode(test_text)
    decoded = preprocessor.decode(encoded)
    print("\n原始文本：", test_text)
    print("编码后：", encoded)
    print("解码后：", decoded)