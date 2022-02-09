import torch.nn as nn


class BertEmbedding(nn.Module):
    """
        1. convert token id to its embedding vector
        2. convert token frequency to its embedding if using bag-of-words
        3. slice/average/summarize subword embedding into the word embedding
        4. apply layerNorm and dropOut
    """
    def __init__(self, manager):
        super().__init__()

        self.hidden_dim = manager.plm_dim



    def forward(self, token_ids):
        return self.embeddings(token_ids)