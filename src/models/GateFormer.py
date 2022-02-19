import torch
from .BaseModel import TwoTowerBaseModel



class TwoTowerGateFormer(TwoTowerBaseModel):
    def __init__(self, manager, newsEncoder, userEncoder, weighter):
        name = "-".join([type(self).__name__, newsEncoder.name, userEncoder.name, f"{manager.enable_gate}_{weighter.name}", str(manager.k)])
        super().__init__(manager, name)

        self.newsEncoder = newsEncoder
        self.userEncoder = userEncoder
        self.weighter = weighter
        self.k = manager.k

        keep_k_modifier = torch.zeros(manager.sequence_length)
        keep_k_modifier[1:self.k + 1] = 1
        self.register_buffer('keep_k_modifier', keep_k_modifier, persistent=False)


    def _compute_gate(self, token_id, attn_mask, gate_mask, token_weight):
        """ gating by the weight of each token

        Returns:
            gated_token_ids: [B, K]
            gated_attn_masks: [B, K]
            gated_token_weight: [B, K]
        """
        if gate_mask is not None:
            keep_k_modifier = self.keep_k_modifier * (gate_mask.sum(dim=-1, keepdim=True) < self.k)
            pad_pos = ~((gate_mask + keep_k_modifier).bool())   # B, L
            token_weight = token_weight.masked_fill(pad_pos, -float('inf'))

            gated_token_weight, gated_token_idx = token_weight.topk(self.k)
            gated_token_weight = torch.softmax(gated_token_weight, dim=-1)
            gated_token_id = token_id.gather(dim=-1, index=gated_token_idx)
            gated_attn_mask = attn_mask.gather(dim=-1, index=gated_token_idx)

        # heuristic gate
        else:
            if token_id.dim() == 2:
                gated_token_id = token_id[:, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, 1: self.k + 1]
            else:
                gated_token_id = token_id[:, :, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, :, 1: self.k + 1]
            gated_token_weight = None

        return gated_token_id, gated_attn_mask, gated_token_weight


    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
            try:
                gate_mask = x['cdd_gate_mask'].to(self.device)
            except:
                # in case that enable_gate is heuristic
                gate_mask = None

        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
            try:
                gate_mask = x['his_gate_mask'].to(self.device)
            except:
                # in case that enable_gate is heuristic
                gate_mask = None

        token_weight = self.weighter(token_id, attn_mask)
        gated_token_id, gated_attn_mask, gated_token_weight = self._compute_gate(token_id, attn_mask, gate_mask, token_weight)
        news_token_embedding, news_embedding = self.newsEncoder(gated_token_id, gated_attn_mask, gated_token_weight)
        return news_token_embedding, news_embedding


    def _encode_user(self, his_news_embedding, his_mask):
        user_embedding = self.userEncoder(his_news_embedding, his_mask=his_mask)
        return user_embedding


    def forward(self, x):
        _, cdd_news_embedding = self._encode_news(x)
        _, his_news_embedding = self._encode_news(x, cdd=False)

        user_embedding = self._encode_user(his_news_embedding, his_mask=x['his_mask'].to(self.device))
        logits = self._compute_logits(cdd_news_embedding, user_embedding)

        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss



