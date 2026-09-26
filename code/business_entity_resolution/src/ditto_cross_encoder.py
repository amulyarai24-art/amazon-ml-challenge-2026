import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List

class DittoTransformerReranker(nn.Module):
    """
    Stage 2 Ditto Transformer Cross-Encoder for deep sequence pair classification.
    Formulates entity pairs into sequence representations:
    [NAME] Company A [ADDR] 123 Main St [SEP] [NAME] Company B [ADDR] 123 Main St
    """
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)

    @staticmethod
    def prepare_sequence_pair(s1_rec: Dict, cand_rec: Dict) -> str:
        s1_str = f"[NAME] {s1_rec.get('name_norm', '')} [ADDR] {s1_rec.get('address_norm', '')} [POSTAL] {s1_rec.get('postal', '')}"
        cand_str = f"[NAME] {cand_rec.get('name_norm', '')} [ADDR] {cand_rec.get('address_norm', '')} [POSTAL] {cand_rec.get('postal', '')}"
        return f"{s1_str} [SEP] {cand_str}"

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep).squeeze(-1)
        return logits