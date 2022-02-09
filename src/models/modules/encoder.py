import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from .attention import scaled_dp_attention


class CnnNewsEncoder(nn.Module):
    def __init__(self, manager):
        super().__init__()

        self.hidden_dim = manager.cnn_dim
        self.embedding_dim = manager.plm_dim

        self.cnn = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1
        )
        nn.init.xavier_normal_(self.cnn.weight)

        self.query_words = nn.Parameter(torch.randn((1, self.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.query_words)
        self.wordQueryProject = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.wordQueryProject.weight)

        self.Tanh = nn.Tanh()
        self.Relu = nn.ReLU()


    def forward(self, news_embedding, attn_mask=None):
        """ encode news through 1-d CNN
        """
        signal_length = news_embedding.shape[-2]
        cnn_input = news_embedding.view(-1, signal_length, self.embedding_dim).transpose(-2, -1)
        cnn_output = self.Relu(self.cnn(cnn_input)).transpose(-2, -1).view(*news_embedding.shape[:-1], self.hidden_dim)

        if attn_mask is not None:
            news_repr = scaled_dp_attention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output, attn_mask.unsqueeze(-2)).squeeze(dim=-2)
        else:
            news_repr = scaled_dp_attention(self.query_words, self.Tanh(self.wordQueryProject(cnn_output)), cnn_output).squeeze(dim=-2)

        return cnn_output, news_repr



class RnnUserEncoder(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.hidden_dim = manager.rnn_dim
        # if manager.encoderU == 'gru':
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        # elif manager.encoderU == 'lstm':
        #     self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)


    def forward(self, news_repr, his_mask=None):
        """
        encode user history into a representation vector

        Args:
            news_repr: batch of news representations, [batch_size, *, hidden_dim]
            news_mask: [batch_size, *, 1]

        Returns:
            user_repr: user representation (coarse), [batch_size, 1, hidden_dim]
        """
        if his_mask is not None:
            lens = his_mask.sum(dim=-1).cpu()
            rnn_input = pack_padded_sequence(news_repr, lens, batch_first=True, enforce_sorted=False)
        else:
            rnn_input = news_repr

        _, user_repr = self.rnn(rnn_input)
        if type(user_repr) is tuple:
            user_repr = user_repr[0]
        return user_repr.transpose(0,1)
