import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """
    A character-level RNN for language modeling.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, type='rnn', num_layers=1, dropout=0.0):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown RNN type: {type}")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output) 
        return output, hidden


class RNN(nn.Module):
    """
    A simple vanilla RNN cell:
        h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + bias)
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.W_xh = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, hidden):
        h_t = torch.tanh(self.W_xh(x) + self.W_hh(hidden) + self.bias)
        return h_t


class LSTM(nn.Module):
    """
    A simple LSTM cell:
        f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  
        i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  
        g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)     
        o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o) 
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.W_f = nn.Linear(embedding_dim, hidden_dim)
        self.W_i = nn.Linear(embedding_dim, hidden_dim)
        self.W_g = nn.Linear(embedding_dim, hidden_dim)
        self.W_o = nn.Linear(embedding_dim, hidden_dim)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        f_t = torch.sigmoid(self.W_f(x) + self.W_hf(h_prev))
        i_t = torch.sigmoid(self.W_i(x) + self.W_hi(h_prev))
        g_t = torch.tanh(self.W_g(x) + self.W_hg(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.W_ho(h_prev))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class CharRNNCustom(nn.Module):
    """
    Character-level language model using custom RNN/LSTM cells.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, type='rnn'):
        super(CharRNNCustom, self).__init__()
        self.hidden_dim = hidden_dim
        self.type = type

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if type == 'rnn':
            self.cell = RNN(embedding_dim, hidden_dim)
        elif type == 'lstm':
            self.cell = LSTM(embedding_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown type: {type}")
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        batch_size = x.size(0)

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            hidden = (h, torch.zeros_like(h)) if self.type == 'lstm' else h

        outputs = []
        for t in range(embedded.size(1)):
            if self.type == 'lstm':
                h, c = self.cell(embedded[:, t, :], hidden)
                hidden = (h, c)
            else:
                h = self.cell(embedded[:, t, :], hidden)
                hidden = h
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        return self.fc(output), hidden