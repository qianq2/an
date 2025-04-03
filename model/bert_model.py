import os
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import BertConfig, BertModel, BertPreTrainedModel
from torch import nn


class Bert_Lite(BertPreTrainedModel):
    def __init__(self, vocab_size, config):
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=config.hidden_dim,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            max_position_embeddings=config.max_length,
            pad_token_id=0
        )
        super().__init__(bert_config)
        self.embeddings = nn.Embedding(vocab_size, config.hidden_dim, padding_idx=0)
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.cls = nn.Linear(config.hidden_dim, vocab_size)
        self._freeze_layers(config.freeze_layers)


    def _freeze_layers(self, num_frozen):
        # 冻结嵌入层
        if num_frozen >= 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        # 冻结编码层
        for layer in self.bert.encoder.layer[:num_frozen]:
            for param in layer.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_mask, labels=None):
        # 词表适配
        inputs_embeds = self.embeddings(input_ids)

        # 预训练BERT处理
        outputs = self.bert(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )

        # 预测得分
        logits = self.cls(outputs.last_hidden_state)

        # 损失计算
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=0.1,  # 标签平滑
                ignore_index=0
            )
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)
