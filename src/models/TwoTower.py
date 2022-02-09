from .BaseModel import TwoTowerBaseModel
from .modules.encoder import *
from .modules.embedding import *



class TwoTowerModel(TwoTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding = BertEmbedding(manager)
        if manager.newsEncoder == "cnn":
            self.newsEncoder = CnnNewsEncoder(manager)
        if manager.userEncoder == "rnn":
            self.userEncoder = RnnUserEncoder(manager)


    def _encode_news(self, token_id, attn_mask):
        news_token_embedding, news_embedding = self.newsEncoder(
            self.embedding(token_id), attn_mask
        )
        return news_token_embedding, news_embedding


    def _encode_user(self, his_news_embedding, his_mask):
        user_embedding = self.userEncoder(his_news_embedding, his_mask=his_mask)
        return user_embedding


    def forward(self, x):
        cdd_token_id = x["cdd_token_id"].to(self.device)
        cdd_attn_mask = x['cdd_attn_mask'].to(self.device)
        _, cdd_news_embedding = self._encode_news(cdd_token_id, cdd_attn_mask)

        his_token_id = x["his_token_id"].to(self.device)
        his_attn_mask = x["his_attn_mask"].to(self.device)
        _, his_news_embedding = self._encode_news(his_token_id, his_attn_mask)

        user_embedding = self._encode_user(his_news_embedding, his_mask=x['his_mask'])
        logits = self._compute_logits(cdd_news_embedding, user_embedding)

        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss

