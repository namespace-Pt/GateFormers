import os
import torch
import logging
import subprocess
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup



class TwoTowerBaseModel(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.his_size = manager.his_size
        self.sequence_length = manager.sequence_length
        self.device = manager.device

        # set all enable_xxx as attributes
        for k,v in vars(manager).items():
            if k.startswith("enable"):
                setattr(self, k, v)
        self.negative_num = manager.negative_num

        name = type(self).__name__
        if manager.verbose is not None:
            self.name = "-".join([name, manager.verbose])
        else:
            self.name = name


    def init_path(self):
        self.encode_dir = os.path.join("data", "cache", "encode", self.name)
        os.makedirs(self.retrieve_dir, exist_ok=True)
        self.logger = logging.getLogger(self.name)


    def get_optimizer(self, manager, dataloader_length):
        optimizer = optim.AdamW(self.parameters(), lr=manager.learning_rate, eps=manager.adam_epsilon)

        scheduler = None
        if manager.scheduler == "linear":
            total_steps = dataloader_length * manager.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = round(manager.warmup * total_steps),
                                            num_training_steps = total_steps)

        return optimizer, scheduler


    def compute_score(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def forward(self,x):
        cdd_repr = self.encode_news(x)
        user_repr, kid = self.encode_user(x)
        score = self.compute_score(cdd_repr, user_repr)

        if self.training:
            logits = nn.functional.log_softmax(score, dim=1)
        else:
            logits = torch.sigmoid(score)

        return logits, kid


    def predict_fast(self, x):
        # [bs, cs, hd]
        cdd_repr = self.news_reprs(x['cdd_id'].to(self.device))
        user_repr, _ = self.encode_user(x)
        scores = self.compute_score(cdd_repr, user_repr)
        logits = torch.sigmoid(scores)
        return logits

