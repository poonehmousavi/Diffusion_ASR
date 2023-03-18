import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import timeit
import random
import datetime


class RnnType:
    GRU = 1
    LSTM = 2

class ActivationFunction:
    RELU = 1
    TANH = 2
    SIGMOID = 3

class Token:
    PAD = 0
    UKN = 1
    SOS = 2
    EOS = 3

class Parameters:

    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))


class RNNEncoder(nn.Module):
    def __init__(self, device, params, embedding):
        super(RNNEncoder, self).__init__()
        self.device = device
        self.params = params
        # Check if valid value for RNN type
        if self.params.rnn_type not in [RnnType.GRU, RnnType.LSTM]:
            raise Exception("Unknown RNN type for encoder. Valid options: {}".format(', '.join([str(t) for t in RnnType])))
        
        # Embedding layer
        self.embedding = embedding
        # RNN layer
        self.num_directions = 2 if self.params.bidirectional_encoder == True else 1
        
        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        
        self.rnn = rnn(self.params.embed_dim,
                       self.params.rnn_hidden_dim,
                       num_layers=self.params.num_layers,
                       bidirectional=self.params.bidirectional_encoder,
                       dropout=self.params.dropout,
                       batch_first=True)
        # Initialize hidden state
        self.hidden = None
        # Define linear layers
        self.linear_dims = params.linear_dims
        # self.linear_dims = [self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states] + self.linear_dims
        # Define last linear output layer
        #self.last_dropout = nn.Dropout(p=self.params.dropout)
        #self.hidden_to_mean = nn.Linear(self.linear_dims[-1], self.params.z_dim)
        #self.hidden_to_logv = nn.Linear(self.linear_dims[-1], self.params.z_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
    
    def init_hidden(self, batch_size):
        if self.params.rnn_type == RnnType.GRU:
            return torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device)
        elif self.params.rnn_type == RnnType.LSTM:
            return (torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))
    
    def forward(self, inputs):
        # inputs.shape = (batch_size, seq_len)
        batch_size, _ = inputs.shape
        # Push through embedding layer ==> X.shape = (batch_size, seq_len, embed_dim)
        X = self.embedding(inputs)
        # Push through RNN layer (the ouput is irrelevant)
        _, self.hidden = self.rnn(X, self.hidden)
        X = self._flatten_hidden(self.hidden, batch_size)
        return X
        # Push trough last linear layer
        #X = self.out(self.last_dropout(X))
        #mean = self.hidden_to_mean(X)
        #logv = self.hidden_to_logv(X)
        #z = self._sample(mean, logv)
        # Return final tensor (will be input for decoder)
        #return mean, self.hidden
        #return mean, logv, z

    def _flatten_hidden(self, h, batch_size):
        if h is None:
            return None
        elif isinstance(h, tuple): # LSTM
            X = torch.cat([self._flatten(h[0], batch_size), self._flatten(h[1], batch_size)], 1)
        else: # GRU
            X = self._flatten(h, batch_size)
        return X


class RNNDecoder(nn.Module):

    def __init__(self, device, params, embedding):
        super(RNNDecoder, self).__init__()
        self.device = device
        self.params = params
        
        # Check if a valid parameter for RNN type is given
        if self.params.rnn_type not in [RnnType.GRU, RnnType.LSTM]:
            raise Exception(
                "Unknown RNN type for encoder. Valid options: {}".format(', '.join([str(t) for t in RnnType])))
        
        # Embedding layer
        self.embedding = embedding
        
        # RNN layer
        self.num_directions = 2 if self.params.bidirectional_encoder == True else 1
        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        self.rnn = rnn(self.params.embed_dim,
                       self.params.rnn_hidden_dim*self.num_directions,
                       num_layers=self.params.num_layers,
                       dropout=self.params.dropout,
                       batch_first=True)
        
        # self.linear_dims = self.params.linear_dims + [self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states]
        #self.z_to_hidden = nn.Linear(self.params.z_dim, self.linear_dims[0])
        # Output layer
        #self.last_dropout = nn.Dropout(p=self.params.dropout)
        # If set, tie weights of output layer to weights of embedding layer
        if self.params.tie_embeddings:
            # Map hidden_dim to embed_dim (can be dropped if hidden_dim=embed_dim)
            self.hidden_to_embed = nn.Linear(self.params.rnn_hidden_dim * self.num_directions, self.params.embed_dim)
            # Weight matrix of self.out has now the same dimension as embedding layer
            self.out = nn.Linear(self.params.embed_dim, self.params.vocab_size)
            # Set weights of output layer to embedding weights. Backprop seems to work quite fine with that.
            self.out.weight.data = self.embedding.embeddings.word_embeddings.weight.permute(1,0)
        else:
            self.out = nn.Linear(self.params.rnn_hidden_dim * self.num_directions, self.params.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, inputs, z, return_outputs=False):
        batch_size, num_steps = inputs.shape
        # "Expand" z vector
        #X = self.z_to_hidden(z)
        X = z
        # Unflatten hidden state for GRU or LSTM
        hidden = self._unflatten_hidden(X, batch_size)
        # Restructure shape of hidden state to accommodate bidirectional encoder (decoder is unidirectional)
        hidden = self._init_hidden_state(hidden)
        # Create SOS token tensor as first input for decoder
        input = torch.LongTensor([[Token.SOS]] * batch_size).to(self.device)
        # Decide whether to do teacher forcing or not
        use_teacher_forcing = random.random() < self.params.teacher_forcing_prob
        # Initiliaze loss
        loss = 0
        outputs = torch.zeros((batch_size, num_steps), dtype=torch.long).to(self.device)
        if use_teacher_forcing:
            for i in range(num_steps):
                output, hidden = self._step(input, hidden)
                topv, topi = output.topk(1)
                outputs[:,i] = topi.detach().squeeze()
                #print(output[0], inputs[:, i][0])
                # loss += self.criterion(output, inputs[:, i])
                input = inputs[:, i].unsqueeze(dim=1)
        else:
            for i in range(num_steps):
                output, hidden = self._step(input, hidden)
                topv, topi = output.topk(1)
                input = topi.detach()
                outputs[:, i] = topi.detach().squeeze()
                #print(topi[0], inputs[:, i][0])
                # loss += self.criterion(output, inputs[:, i])
                if input[0].item() == Token.EOS:
                    break
        # Return loss
        if return_outputs == True:
            return  outputs
        else:
            return loss

    def _init_hidden_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        elif isinstance(encoder_hidden, tuple): # LSTM
            return tuple([self._concat_directions(h) for h in encoder_hidden])
        else: # GRU
            return self._concat_directions(encoder_hidden)

    def _concat_directions(self, hidden):
        # hidden.shape = (num_layers * num_directions, batch_size, hidden_dim)
        #print(hidden.shape, hidden[0:hidden.size(0):2].shape)
        if self.params.bidirectional_encoder:
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            # Alternative approach (same output but easier to understand)
            #h = hidden.view(self.params.num_layers, self.num_directions, hidden.size(1), self.params.rnn_hidden_dim)
            #h_fwd = h[:, 0, :, :]
            #h_bwd = h[:, 1, :, :]
            #hidden = torch.cat([h_fwd, h_bwd], 2)
        return hidden

    def _step(self, input, hidden):
        # Get embedding of current input word:
        X = self.embedding(input)
        # Push input word through rnn layer with current hidden state
        output, hidden = self.rnn(X, hidden)
        # Push output through linear layer to get to vocab_size

        if self.params.tie_embeddings == True:
            output = F.log_softmax(self.out(self.hidden_to_embed(output.squeeze(dim=1))), dim=1)
        else:
            #output = F.log_softmax(self.out(self.last_dropout(output.squeeze(dim=1))), dim=1)
            output = F.log_softmax(self.out(output.squeeze(dim=1)), dim=1)
        # return the output (batch_size, vocab_size) and new hidden state
        return output, hidden

    def _unflatten_hidden(self, X, batch_size):
        if X is None:
            return None
        elif self.params.rnn_type == RnnType.LSTM:  # LSTM
            X_split = torch.split(X, int(X.shape[1]/2), dim=1)
            h = (self._unflatten(X_split[0], batch_size), self._unflatten(X_split[1], batch_size))
        else:  # GRU
            h = self._unflatten(X, batch_size)
        return h

    def _unflatten(self, X, batch_size):
        # (batch_size, num_directions*num_layers*hidden_dim)    ==>
        # (batch_size, num_directions * num_layers, hidden_dim) ==>
        # (num_layers * num_directions, batch_size, hidden_dim) ==>
        return X.view(batch_size, self.params.num_layers * self.num_directions, self.params.rnn_hidden_dim).transpose(0, 1).contiguous()

class RNN_AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(RNN_AE,self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x, decoder_inputs):
        z = self.encoder(x)
        return self.decoder(decoder_inputs, z)



