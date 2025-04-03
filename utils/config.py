
# # 训练参数
# BATCH_SIZE = 256
# SEQ_LENGTH = 50  # 与预处理参数一致
# EPOCHS = 50
# LEARNING_RATE = 0.001
#
# # 模型配置
# EMBEDDING_DIM = 128  # 增大嵌入维度（适配中文字形多样性）
# HIDDEN_DIM = 256     # 增加LSTM隐藏层维度
# NUM_LAYERS = 1
#
# # 正则化配置
# DROPOUT_RATE = 0.3
# WEIGHT_DECAY = 1e-5
import torch

class Config:
    # 数据参数
    file_path = "data/西游记.txt"          # 输入文本路径
    max_length = 256                      # 最大序列长度（提升文本建模能力）
    min_freq = 2                          # 字符最小出现频次

    # 模型参数
    hidden_dim = 256                        # 隐藏层维度（平衡容量和显存）
    num_layers = 6                          # Transformer层数（恢复标准结构）
    num_heads = 2                           # 注意力头数（保持计算效率）
    freeze_layers = 6                       # 冻结前6层（加速训练）


    # 训练参数
    batch_size = 64                         # 实际批次大小（经梯度累积等效24）
    learning_rate = 2e-5                    # 学习率
    epochs = 300                            # 训练轮次
    trained_model_patch = "model/trained_model/bert_model.pth"      # 模型保存路径


    # 早停法参数
    patience = 4

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")