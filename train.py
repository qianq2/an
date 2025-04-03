import torch
from utils.config import Config
from model.bert_model import Bert_Lite
from utils.utils import TextPreprocessor
from torch.utils.data import DataLoader, random_split
#from model.model import CharLSTM


class Trainer:
    def __init__(self):
        self.config = Config()
        self.preprocessor = TextPreprocessor(self.config)
        self.model = Bert_Lite(vocab_size=self.preprocessor.vocab_size,
                               config=self.config).to(self.config.device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p:p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate
        )
        self.scaler = torch.cuda.amp.GradScaler()

    def create_loader(self):
        dataset = self.preprocessor.create_sequences()
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        return (
            DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for x, mask, y in train_loader:
            x, mask, y = x.to(self.config.device), mask.to(self.config.device), y.to(self.config.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss, _ = self.model(input_ids=x,
                                     attention_mask=mask,
                                     labels=y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask, y = x.to(self.config.device), mask.to(self.config.device), y.to(self.config.device)
                loss, _ = self.model(x, mask, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)


    def run(self):
        train_loader, val_loader = self.create_loader()
        best_loss = float('inf')
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.trained_model_patch)
                print("模型已保存！")

            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"早停，第{epoch + 1}轮结束训练。")
                    break
        print("训练完成！最佳模型已保存至 model/trained_model/Bert_model.pth")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()