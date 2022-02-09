from .BaseModel import TwoTowerBaseModel



class TwoTowerModel(TwoTowerBaseModel):
    def __init__(self, manager, newsEncoder, userEncoder):
        super().__init__(manager, name="-".join(["TwoTower", newsEncoder.name, userEncoder.name]))
        self.newsEncoder = newsEncoder
        self.userEncoder = userEncoder


    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
        news_token_embedding, news_embedding = self.newsEncoder(token_id, attn_mask)
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

