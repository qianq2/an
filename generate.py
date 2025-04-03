
from model.bert_model import Bert_Lite
from utils.utils import TextPreprocessor
from utils.config import Config
import torch
import re

class Generate_text:
    def __init__(self, model, preprocessor, config):

        self.model =model
        self.preprocessor = preprocessor
        self.config = config
        self.model.eval()   # 设置为评估模式，关闭dropout等训练专用层

    def _prepare_input(self, text):
        """ 输入预处理管道（保持与训练时一致的清洗逻辑）"""
        # 保持与训练完全相同的字符过滤规则
        kept_chars = r'\u4e00-\u9fa5，。！？、；：“”‘’（）【】…—～《》\n'
        clean_text = re.sub(f'[^{kept_chars}0-9]', '', text)
        # 数字标准化处理
        clean_text = re.sub(r'\d+', '#NUM', clean_text)

        # 编码流程：添加CLS标记 + 实际编码
        encoded = [self.preprocessor.char2idx['<CLS>']]      # 起始标记
        encoded += self.preprocessor.encode(clean_text)      # 主体内容编码
        return encoded

    def generate(self, prompt, max_length=100, temperature=0.8, top_k=5):
        """文本生成函数"""
        current_ids = self._prepare_input(prompt)
        sep_id = self.preprocessor.char2idx.get('<SEP>', -1)

        for _ in range(max_length):
            # 截断到模型最大长度
            input_ids = current_ids[-(self.config.max_length - 1):]
            attention_mask = [1] * len(input_ids)

            # 转换为模型输入张量
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            mask_tensor = torch.tensor([attention_mask], dtype=torch.long)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(input_tensor, mask_tensor)

            # 获取最后一个位置的logits并进行温度调节
            logits = outputs[0][0, -1, :]  # 获取最后一个位置的logits
            logits = logits / temperature

            # Top-k筛选
            if top_k > 0:
                topk = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits[topk.indices] = topk.values

            # 概率采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # 更新生成序列
            current_ids.append(next_token)
            if next_token == sep_id:
                break

        # 解码并去除特殊标记
        generated = []
        for idx in current_ids:
            if idx == self.preprocessor.char2idx['<CLS>']:
                continue
            if idx == sep_id:
                break
            generated.append(idx)

        return self.preprocessor.decode(generated)


if __name__ == "__main__":
    config = Config()
    preprocessor = TextPreprocessor(config)

    model = Bert_Lite(preprocessor.vocab_size, config)
    model.load_state_dict(torch.load("model/trained_model/best_model.pth"))

    # 创建生成器
    generator = Generate_text(model, preprocessor, config)
    # print("模型参数校验:", sum(p.numel() for p in model.parameters()))
    # 生成示例
    prompt = "悟空"
    generated_text = generator.generate(
        prompt,
        max_length=100,
        temperature=0.8,
        top_k=50
    )

    print(f"Prompt: {prompt}")
    print(f"生成语句: {generated_text}")

