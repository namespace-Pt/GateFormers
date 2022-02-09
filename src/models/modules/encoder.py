import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence
from .attention import scaled_dp_attention, TFMLayer



class BaseNewsEncoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.sequence_length = manager.sequence_length
        self.name = type(self).__name__[:-11]



class BaseUserEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__[:-11]



class CnnNewsEncoder(BaseNewsEncoder):
    def __init__(self, manager):
        super().__init__(manager)

        self.embedding_dim = manager.plm_dim
        bert = AutoModel.from_pretrained(manager.plm_dir)
        self.embedding = bert.embeddings.word_embeddings

        self.cnn = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=manager.hidden_dim,
            kernel_size=3,
            padding=1
        )
        nn.init.xavier_normal_(self.cnn.weight)

        self.query_words = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.query_words)
        self.wordQueryProject = nn.Linear(manager.hidden_dim, manager.hidden_dim)
        nn.init.xavier_normal_(self.wordQueryProject.weight)

        self.Tanh = nn.Tanh()
        self.Relu = nn.ReLU()


    def forward(self, token_id, attn_mask=None):
        """ encode news through 1-d CNN
        """
        token_embeddings = self.embedding(token_id)
        cnn_input = token_embeddings.view(-1, self.sequence_length, self.embedding_dim).transpose(-2, -1)
        cnn_output = self.Relu(self.cnn(cnn_input)).transpose(-2, -1).view(*token_embeddings.shape[:-1], -1)

        if attn_mask is not None:
            news_embedding = scaled_dp_attention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output, attn_mask.unsqueeze(-2)).squeeze(dim=-2)
        else:
            news_embedding = scaled_dp_attention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output).squeeze(dim=-2)

        return cnn_output, news_embedding



class BertNewsEncoder(BaseNewsEncoder):
    def __init__(self, manager):
        super().__init__(manager)
        self.plm = AutoModel.from_pretrained(manager.plm_dir)
        self.plm.pooler = None


    def forward(self, token_id, attn_mask):
        original_shape = token_id.shape
        token_id = token_id.view(-1, self.sequence_length)
        attn_mask = attn_mask.view(-1, self.sequence_length)

        token_embeddings = self.plm(token_id, attention_mask=attn_mask).last_hidden_state
        news_embedding = token_embeddings[:, 0].view(*original_shape[:-1], -1)
        token_embeddings = token_embeddings.view(*original_shape, -1)
        return token_embeddings, news_embedding



class TfmNewsEncoder(BaseNewsEncoder):
    def __init__(self, manager):
        super().__init__(manager)
        self.embedding_dim = manager.plm_dim
        bert = AutoModel.from_pretrained(manager.plm_dir)
        self.embedding = bert.embeddings.word_embeddings

        self.transformer = TFMLayer(manager)


    def forward(self, token_id, attn_mask):
        original_shape = token_id.shape
        token_id = token_id.view(-1, self.sequence_length)
        attn_mask = attn_mask.view(-1, self.sequence_length)

        token_embeddings = self.transformer(self.embedding(token_id), attention_mask=attn_mask)
        news_embedding = token_embeddings[:, 0].view(*original_shape[:-1], -1)
        token_embeddings = token_embeddings.view(*original_shape, -1)
        return token_embeddings, news_embedding



class RnnUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()
        # if manager.encoderU == 'gru':
        self.rnn = nn.GRU(manager.hidden_dim, manager.hidden_dim, batch_first=True)
        # elif manager.encoderU == 'lstm':
        #     self.rnn = nn.LSTM(manager.hidden_dim, manager.hidden_dim, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)


    def forward(self, news_embedding, his_mask=None):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        if his_mask is not None:
            lens = his_mask.sum(dim=-1).cpu()
            rnn_input = pack_padded_sequence(news_embedding, lens, batch_first=True, enforce_sorted=False)
        else:
            rnn_input = news_embedding

        _, user_embedding = self.rnn(rnn_input)
        if type(user_embedding) is tuple:
            user_embedding = user_embedding[0]
        return user_embedding.transpose(0,1)



class SumUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()


    def forward(self, news_embedding, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_embedding = news_embedding.sum(dim=-2, keepdim=True)
        return user_embedding



class AvgUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()


    def forward(self, news_embedding, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_embedding = news_embedding.mean(dim=-2, keepdim=True)
        return user_embedding



class AttnUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()

        self.user_query = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.user_query)



    def forward(self, news_embedding, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_embedding = scaled_dp_attention(self.user_query, news_embedding, news_embedding)
        return user_embedding



class TfmUserEncoder(BaseUserEncoder):
    def __init__(self, manager):
        super().__init__()
        self.transformer = TFMLayer(manager)


    def forward(self, news_embedding, his_mask, **kargs):
        """
        encode user history into a representation vector

        Args:
            news_embedding: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_embedding: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        user_embedding = self.transformer(news_embedding, attention_mask=his_mask)[:, [0]]
        return user_embedding

