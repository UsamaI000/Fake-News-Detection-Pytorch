# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

# LSTM MODEL CLASS
class LSTMSentimentNet(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings, bi=False):
        super(LSTMSentimentNet, self).__init__()
        self.config = config  
        self.output_size = self.config.output_size
        self.n_layers = self.config.hidden_layers
        self.hidden_dim = self.config.hidden_size
        self.embedding_dim = self.config.embed_size
        self.drop_prob = self.config.dropout_keep
        self.bi_direction = bi
        
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, dropout=self.drop_prob, batch_first=True, bidirectional=self.bi_direction)
        self.dropout = nn.Dropout(self.drop_prob)
        if self.bi_direction:   # Check if Bi-directional 
          self.fc = nn.Linear(self.hidden_dim*2, self.output_size)
          print('Using Bi-Directional LSTM')
        else: 
          self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward (self, input_words):
        input_words = input_words.long()                # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)         # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        return out, h


# RCNN MODEL CLASS
class RCNNSentimentNet(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(RCNNSentimentNet, self).__init__()
        self.config = config 
        self.output_size = self.config.output_size
       
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=True)        
        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(input_size = self.config.embed_size,
                            hidden_size = self.config.hidden_size,
                            num_layers = self.config.hidden_layers,
                            dropout = self.config.dropout_keep,
                            bidirectional = True,
                            batch_first=True)        
        self.dropout = nn.Dropout(self.config.dropout_keep)        
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(
            self.config.embed_size + 2*self.config.hidden_size,
            self.config.hidden_size_linear)        
        
        # Tanh non-linearity
        self.tanh = nn.Tanh()         
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size_linear,
            self.output_size)
        
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = x.T
        # x.shape = (seq_len, batch_size)
        embedded_sent = self.embeddings(x)
       
        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)                         # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)
        input_features = torch.cat([lstm_out,embedded_sent], 2).permute(1,0,2) # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
       
        linear_output = self.tanh(self.W(input_features))                      # linear_output.shape = (batch_size, seq_len, hidden_size_linear)
        linear_output = linear_output.permute(0,2,1)                           # Reshaping fot max_pool
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)    # max_out_features.shape = (batch_size, hidden_size_linear)
        
        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        return final_out

