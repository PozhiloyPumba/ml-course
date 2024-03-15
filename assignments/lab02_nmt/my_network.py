import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, dropout=0.2):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
         
    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout : float, posEnc : bool = False, attention: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.posEnc = None
        if posEnc:
            self.posEnc = PositionalEncoding(emb_dim, 256, dropout)

        self.attention = None

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
            # <YOUR CODE HERE>
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True
        )
            # <YOUR CODE HERE>
        
        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>
        
    def forward(self, src, hidden=None):
        
        embedded = self.dropout(self.embedding(src))

        if not hidden:
            hidden = torch.randn(2 * self.n_layers, src.shape[-1], self.hid_dim)
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        output, hidden = self.rnn(embedded, hidden)
        output = output[:, :, :self.hid_dim] + output[:, :, self.hid_dim:]

        hidden = torch.tanh(hidden)
        
        return output, hidden
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout : float, posEnc : bool = False, attention: bool = False):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.attention = None
        self.posEnc = None
        if attention:
            self.attention1 = Attention(hid_dim)

            if posEnc:
                self.posEnc = PositionalEncoding(hid_dim, 256, dropout)

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.GRU(
            input_size=emb_dim if self.attention is None else hid_dim + emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=(0 if n_layers == 1 else dropout)
        )
            # <YOUR CODE HERE>
        
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
            # <YOUR CODE HERE>
        
        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input).unsqueeze(0))
        if self.posEnc is not None:
            encoder_outputs = self.posEnc(encoder_outputs)

        rnn_input = embedded
        
        if self.attention is not None:
            key = hidden.sum(axis=0)
            attn_weights = self.attention(key, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            context = context.transpose(0, 1)
            rnn_input = torch.cat([embedded, context], -1)

        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(0)

        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_output, hidden = self.encoder(src)
        hidden = hidden[-self.decoder.n_layers:]
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
