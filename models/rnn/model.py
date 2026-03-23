import torch
import torch.nn as nn

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

class CharRNN(nn.Module):
    """
    A character-level RNN for language modeling.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output) 
        return output, hidden