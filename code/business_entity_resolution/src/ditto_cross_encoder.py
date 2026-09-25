import torch.nn as nn

class DittoTransformerReranker(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.model_name = model_name

    def prepare_sequence_pair(self, s1_rec: dict, cand_rec: dict) -> str:
        return f"[NAME] {s1_rec.get('name_norm','')} [SEP] [NAME] {cand_rec.get('name_norm','')}"