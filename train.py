import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.model import CharLSTM
from utils import TextPreprocessor
from config import *

def main():
    # 初始化预处理器
    preprocessor = TextPreprocessor(
        file_path="data/西游记.txt",
        seq_length=SEQ_LENGTH,
        min_freq=2
    )
    sequences, targets = preprocessor.create_sequences()

    # 划分训练集和验证集（9:1）
    dataset_size = len(sequences)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(sequences, targets), [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4  # 多线程加速数据加载
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 初始化模型、优化器和损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(  # 新增学习率调度
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # 混合精度训练（可选）
    use_amp = True  # 若GPU支持则设为True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 训练循环
    best_val_loss = float('inf')
    #早停法
    patience = 3  # 允许验证损失不下降的轮次
    patience_counter = 0
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs, _ = model(x)
                loss = criterion(outputs, y)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs, _ = model(x)
                val_loss += criterion(outputs, y).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 学习率调度
        scheduler.step(avg_val_loss)  # 根据验证损失调整学习率

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model/best_model.pth")
            print("模型已保存！")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停，第{epoch+1}轮结束训练。")
                break

    print("训练完成！最佳模型已保存至 model/best_model.pth")

if __name__ == "__main__":
    main()