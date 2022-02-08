import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .TwoTowerBaseModel import TwoTowerBaseModel
from .Encoders.BERT import BERT_Encoder
from models.UniLM.configuration_tnlrv3 import TuringNLRv3Config
from models.UniLM.modeling import TuringNLRv3ForSequenceClassification, relative_position_bucket

class PLM(TwoTowerBaseModel):
    def __init__(self, manager, encoderU):
        # Pretrained language model naive baseline
        super().__init__(manager)

        self.encoderU = encoderU

        if manager.debias:
            self.userBias = nn.Parameter(torch.randn(1,manager.bert_dim))
            nn.init.xavier_normal_(self.userBias)

        if manager.bert == 'unilm':
            config = TuringNLRv3Config.from_pretrained(manager.unilm_config_path)
            bert = TuringNLRv3ForSequenceClassification.from_pretrained(manager.unilm_path, config=config).bert

        elif manager.bert == 'deberta':
            # add a pooler
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )
            self.pooler = nn.Sequential(
                nn.Linear(manager.bert_dim, manager.bert_dim),
                nn.GELU()
            )

        elif manager.bert == "funnel":
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )
            self.pooler = do_nothing

        elif manager.bert == "synthesizer":
            from transformers import BertConfig, BertModel
            from .Modules.Synthesizer import BertSelfAttention
            bert_config = BertConfig()
            # primary bert
            bert = BertModel(bert_config)
            # [CLS]
            bert_config.signal_length = self.signal_length
            for l in bert.encoder.layer:
                l.attention.self = BertSelfAttention(bert_config)
            bert.load_state_dict(BertModel.from_pretrained("bert-base-uncased", cache_dir=manager.path + 'bert_cache/').state_dict(), strict=False)

        elif manager.bert == "distill":
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )
            self.pooler = nn.Sequential(
                nn.Linear(manager.bert_dim, manager.bert_dim),
                nn.GELU()
            )

        elif manager.bert == "newsbert":
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )
            bert.encoder.layer = bert.encoder.layer[:4]

        elif manager.bert == "longformer":
            from transformers import LongformerModel, LongformerConfig
            bert_config = LongformerConfig()
            bert_config.attention_window = 32
            bert_config.vocab_size = 50265
            bert = LongformerModel(bert_config)

        elif manager.bert == "bigbird":
            from transformers import BigBirdModel, BigBirdConfig
            bert_config = BigBirdConfig()
            bert_config.block_size = 64
            # bert_config.num_random_block = 0
            bert = BigBirdModel(bert_config)

        else:
            bert = AutoModel.from_pretrained(
                manager.get_bert_for_load(),
                cache_dir=manager.path + 'bert_cache/'
            )

        self.bert = bert

        self.name = '__'.join(["plm", manager.bert, manager.encoderU])


    def encode_news(self, x):
        """
        encode news of loader_news
        """
        # encode news with MIND_news
        batch_size = x['cdd_encoded_index'].size(0)
        cdd_news = x["cdd_encoded_index"].to(self.device).view(-1, self.signal_length)
        cdd_attn_mask = x['cdd_attn_mask'].to(self.device).view(-1, self.signal_length)

        cdd_news_repr = self.bert(cdd_news, cdd_attn_mask)[-1]
        if hasattr(self, 'pooler'):
            cdd_news_repr = self.pooler(cdd_news_repr[:, 0])
            # cdd_news_repr = cdd_news_repr[:, 0]
        cdd_news_repr = cdd_news_repr.view(batch_size, -1, self.hidden_dim)
        return cdd_news_repr


    def encode_user(self, x):
        # fast encoding user
        if hasattr(self, "news_reprs"):
            his_news_repr = self.news_reprs(x['his_id'].to(self.device))

        # slow encoding user, PLM per historical piece
        else:
            his_news = x["his_encoded_index"].to(self.device)
            his_attn_mask = x["his_attn_mask"].to(self.device)
            batch_size = his_news.size(0)

            his_news_repr = self.bert(his_news, his_attn_mask)[-1]

            if hasattr(self, 'pooler'):
                his_news_repr = self.pooler(his_news_repr[:, 0])
                # his_news_repr = his_news_repr[:, 0]
            his_news_repr = his_news_repr.view(batch_size, self.his_size, self.hidden_dim)

        user_repr = self.encoderU(his_news_repr, his_mask=x['his_mask'])
        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return user_repr, None
