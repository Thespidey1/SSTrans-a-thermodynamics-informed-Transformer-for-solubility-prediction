from transformers import DataCollatorForLanguageModeling
import torch
from typing import Optional, List, Dict


class SafeDimensionDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 强制转换输入为合法批次格式
        batch = {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples])
        }

        # 执行掩码操作
        masked_inputs, labels = self.torch_mask_tokens(batch)

        # 重构符合父类预期的输出格式
        return {"input_ids": masked_inputs, "labels": labels}

    def torch_mask_tokens(self, inputs: Dict[str, torch.Tensor],
                          special_tokens_mask: Optional[torch.Tensor] = None) -> tuple:
        # 维度验证（关键修复点）
        input_ids = inputs["input_ids"]
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids 必须是严格的二维张量，当前维度：{input_ids.dim()}（建议检查数据预处理）")

        # 安全生成掩码（完整实现）
        attention_mask = inputs.get("attention_mask", None)
        labels = input_ids.clone()

        # 概率矩阵生成（兼容二维）
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability, device=input_ids.device)

        # 填充位置过滤（基于attention_mask）
        if attention_mask is not None:
            padding_mask = attention_mask.bool()
            probability_matrix *= padding_mask.float()

        # 特殊标记过滤
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in input_ids.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=input_ids.device)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        # 生成最终掩码
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 掩码替换逻辑（严格二维操作）
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
        input_ids = input_ids.masked_fill(indices_replaced, self.tokenizer.mask_token_id)

        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long, device=input_ids.device)
        input_ids = torch.where(indices_random, random_words, input_ids)

        return input_ids, labels
