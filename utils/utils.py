import re
import torch
from collections import Counter
from torch.utils.data import TensorDataset



class TextPreprocessor:
    def __init__(self, config):
        """初始化预处理实例"""
        self.config = config
        self._load_and_clean()
        self._build_vocab()

    def _load_and_clean(self):
        """加载并清洗原始文本"""
        # 保留中文字符、标点和数字
        with open(self.config.file_path, 'r', encoding='UTF-8') as f:
            text = f.read()
        kept_chars = r'\u4e00-\u9fa5，。！？、；：“”‘’（）【】…—～《》\n'
        # 清洗非中文字符
        self.clean_text = re.sub(f'[^{kept_chars}0-9]', '', text)
        # 统一数字表示
        self.clean_text = re.sub(r'\d+', '#NUM', self.clean_text)

    def _build_vocab(self):
        """构建字符到索引的映射词典"""
        char_counts = Counter(self.clean_text)
        # 基础词表：特殊符号+高频字符
        self.vocab = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']       # 特殊符号
        self.vocab += [char for char, cnt in char_counts.items() if cnt >= self.config.min_freq]

        # 创建双向映射
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}

    def encode(self, text):
        """编码文本为索引"""
        return [self.char2idx.get(c, 1) for c in text]  # 1=<UNK>

    def decode(self, indices):
        """将索引列表转换回文本"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])


    def create_sequences(self):
        """生成训练序列"""
        # 编码并添加起止符
        encoded = [self.char2idx['<CLS>']] + self.encode(self.clean_text) + [self.char2idx['<SEP>']]
        seqs, masks, labels = [], [], []

        # 滑动窗口生成序列
        step = self.config.max_length
        for i in range(0, len(encoded) - self.config.max_length, step):
            chunk = encoded[i:i + self.config.max_length]
            seq = chunk[:-1]  # 输入序列
            target = chunk[1:]  # 目标序列
            mask = [1 if t != 0 else 0 for t in seq]

            seqs.append(seq)
            masks.append(mask)
            labels.append(target)

        return TensorDataset(
            torch.tensor(seqs),
            torch.tensor(masks),
            torch.tensor(labels))

    @property
    def vocab_size(self):
        return len(self.vocab)

