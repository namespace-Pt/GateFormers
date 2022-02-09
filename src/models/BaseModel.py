import os
import math
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
from utils.util import pack_results, compute_metrics



class BaseModel(nn.Module):
    def __init__(self, manager, name=None):
        super().__init__()

        self.his_size = manager.his_size
        self.sequence_length = manager.sequence_length
        self.device = manager.device
        self.rank = manager.rank
        self.world_size = manager.world_size

        # set all enable_xxx as attributes
        for k,v in vars(manager).items():
            if k.startswith("enable"):
                setattr(self, k, v)
        self.negative_num = manager.negative_num

        if name is None:
            name = type(self).__name__
        if manager.verbose is not None:
            self.name = "-".join([name, manager.verbose])
        else:
            self.name = name

        self.crossEntropy = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(self.name)


    def get_optimizer(self, manager, dataloader_length):
        optimizer = optim.AdamW(self.parameters(), lr=manager.learning_rate)

        scheduler = None
        if manager.scheduler == "linear":
            total_steps = dataloader_length * manager.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = round(manager.warmup * total_steps),
                                            num_training_steps = total_steps)

        return optimizer, scheduler


    def _gather_tensors(self, local_tensor):
        """
        gather tensors from all gpus

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            all_tensors: concatenation of local_tensor in each process
        """
        all_tensors = [torch.empty_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, local_tensor)
        all_tensors[self.rank] = local_tensor
        return torch.cat(all_tensors, dim=0)




class TwoTowerBaseModel(BaseModel):
    def __init__(self, manager):
        super().__init__(manager)


    def _compute_logits(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def encode_news(self, manager, loader_news):
        news_embeddings = torch.zeros((len(loader_news.sampler), self.newsEncoder.hidden_dim), device=self.device)

        start_idx = end_idx = 0
        for i, x in enumerate(tqdm(loader_news, ncols=80, desc="Encoding News")):
            cdd_token_id = x["cdd_token_id"].to(self.device)
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)
            _, news_embedding = self._encode_news(cdd_token_id, cdd_attn_mask)

            end_idx = start_idx + news_embedding.shape[0]
            news_embeddings[start_idx: end_idx] = news_embedding
            start_idx = end_idx

        self.news_embeddings = self._gather_tensors(news_embeddings)


    def _dev(self, manager, loader):
        impr_indices = []
        masks = []
        labels = []
        preds = []

        for i, x in enumerate(tqdm(loader, ncols=80, desc="Predicting")):
            cdd_idx = x["cdd_idx"].to(self.device, non_blocking=True)
            his_idx = x["his_idx"].to(self.device, non_blocking=True)
            cdd_embedding = self.news_embeddings[cdd_idx]
            his_embedding = self.news_embeddings[his_idx]
            user_embedding = self._encode_user(his_embedding, his_mask=x['his_mask'])
            logits = self._compute_logits(cdd_embedding, user_embedding)

            masks.extend(x["cdd_mask"].tolist())
            impr_indices.extend(x["impr_index"].tolist())
            labels.extend(x["label"].tolist())
            preds.extend(logits.tolist())

        if manager.distributed:
            dist.barrier(device_ids=[self.device])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indices, masks, labels, preds))

            if self.rank == 0:
                impr_indices = []
                masks = []
                labels = []
                preds = []
                for output in outputs:
                    impr_indices.extend(output[0])
                    masks.extend(output[1])
                    labels.extend(output[2])
                    preds.extend(output[3])

                masks = np.asarray(masks, dtype=np.bool8)
                labels = np.asarray(labels, dtype=np.int32)
                preds = np.asarray(preds, dtype=np.float32)

                labels, preds = pack_results(impr_indices, masks, labels, preds)

        return labels, preds


    def dev(self, manager, loaders, load=True, log=False):
        self.eval()
        if load:
            manager.load(self)

        self.encode_news(manager, loaders["news"])
        labels, preds = self._dev(manager, loaders["dev"])

        if self.rank in [0, -1]:
            metrics = compute_metrics(labels, preds, manager.metrics)
            metrics["main"] = metrics["auc"]
            self.logger.info(metrics)
            if log:
                manager._log(self.name, metrics)
        else:
            metrics = None

        if manager.distributed:
            dist.barrier(device_ids=[self.device])

        return metrics