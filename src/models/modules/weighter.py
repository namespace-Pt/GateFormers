import torch
import torch.nn as nn
from transformers import AutoModel
from .attention import TFMLayer


class BaseWeighter(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = type(self).__name__[:-8]

        self.weightPooler = nn.Sequential(
            nn.Linear(manager.gate_hidden_dim, manager.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(manager.dropout_p),
            nn.Linear(manager.gate_hidden_dim, 1)
        )


    def _compute_weight(self, embeddings):
        weights = self.weightPooler(embeddings).squeeze(-1)
        return weights



class AllBertWeighter(BaseWeighter):
    def __init__(self, manager):
        super().__init__(manager)

        self.bert = AutoModel.from_pretrained(manager.plm_dir)
        self.bert.pooler = None


    def forward(self, token_ids, attn_masks):
        """
        Args:
            token_ids: [B, L]
            attn_masks: [B, L]

        Returns:
            weights: [B, L]
        """
        bert_embeddings = self.bert(token_ids, attention_mask=attn_masks)[0]    # B, L, D
        weights = self._compute_weight(bert_embeddings)
        return weights



class CNNWeighter(BaseWeighter):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding = nn.Embedding(manager.vocab_size, manager.gate_embedding_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=manager.gate_embedding_dim,
                out_channels=manager.gate_hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.cnn[0].weight)


    def forward(self, token_ids, attn_masks):
        """
        Args:
            token_ids: [B, L]
            attn_masks: [B, L]

        Returns:
            weights: [B, L]
        """
        embeddings = self.embedding(token_ids)    # B, L, D
        conv_embeddings = self.cnn(embeddings.transpose(-1, -2)).transpose(-1, -2)
        weights = self._compute_weight(conv_embeddings)
        return weights



class TFMWeighter(BaseWeighter):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding = nn.Embedding(manager.vocab_size, manager.gate_hidden_dim)
        self.tfm = TFMLayer(manager)


    def forward(self, token_ids, attn_masks):
        embeddings = self.embedding(token_ids)    # B, L, D
        tfm_embeddings = self.tfm(embeddings, attention_masks=attn_masks)
        weights = self._compute_weight(tfm_embeddings)
        return weights



class FirstWeighter(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.name = "First"

    def forward(self, token_ids, attn_masks):
        weights = torch.arange(1, 0, -1 / token_ids.size(-1), dtype=torch.float, device=token_ids.device)
        weights = weights.unsqueeze(0).expand(token_ids.shape)
        return weights