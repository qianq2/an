# 训练参数
BATCH_SIZE = 256
SEQ_LENGTH = 50  # 与预处理参数一致
EPOCHS = 50
LEARNING_RATE = 0.001

# 模型配置
EMBEDDING_DIM = 128  # 增大嵌入维度（适配中文字形多样性）
HIDDEN_DIM = 256     # 增加LSTM隐藏层维度
NUM_LAYERS = 1

# 正则化配置
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-5
