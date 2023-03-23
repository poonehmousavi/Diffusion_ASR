import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np



class Parameters:

    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))

def pack_padded_sequence(inputs, lengths):
    """Returns packed speechbrain-formatted tensors.

    Arguments
    ---------
    inputs : torch.Tensor
        The sequences to pack.
    lengths : torch.Tensor
        The length of each sequence.
    """
    return torch.nn.utils.rnn.pack_padded_sequence(
        inputs, lengths, batch_first=True, enforce_sorted=False
    )


def pad_packed_sequence(inputs):
    """Returns speechbrain-formatted tensor from packed sequences.

    Arguments
    ---------
    inputs : torch.nn.utils.rnn.PackedSequence
        An input set of sequences to convert to a tensor.
    """
    outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        inputs, batch_first=True
    )
    return outputs


class RNNEncoder(nn.Module):
    def __init__(self, device, embedding, embedding_size, rnn_type, hidden_size, latent_size,
                dropout, num_layers=1, bidirectional=False, ):
        super(RNNEncoder, self).__init__()
        self.device = device


        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise Exception("Unknown RNN type for encoder. Valid options: rnn, gru, lstm .")
        
        # Embedding layer
        self.embedding = embedding
        
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        
        
        self.rnn = rnn( input_size=self.embedding_size,
                       hidden_size=self.hidden_size,
                       num_layers= self.num_layers,
                       bidirectional=self.bidirectional,
                       dropout=self.dropout,
                       batch_first=True)
        

        self.hidden2latent = nn.Linear(hidden_size * self.hidden_factor,latent_size)

        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
    
    # def init_hidden(self, batch_size,):
    #     if self.params.rnn_type == RnnType.GRU:
    #         return torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device)
    #     elif self.params.rnn_type == RnnType.LSTM:
    #         return (torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
    #                 torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))
    
    def forward(self, inputs,lengths=None):

        # Flatten params for data parallel
        self.rnn.flatten_parameters()

        # inputs.shape = (batch_size, seq_len)
        batch_size, seq_len = inputs.shape
        # Push through embedding layer ==> X.shape = (batch_size, seq_len, embed_dim)
        x = self.embedding(inputs).last_hidden_state

        # Pack sequence for proper RNN handling of padding
        if lengths is not None:
            x = pack_padded_sequence(x, lengths)
        # Push through RNN layer
        output ,h_n  = self.rnn(x)
        hidden = self._flatten_hidden(h_n, batch_size)
        #  Unpack the packed sequence
        if lengths is not None:
            output = pad_packed_sequence(output)
        latent = self.hidden2latent(hidden)
        return output,h_n,latent
    

    
    def _flatten(self, h, batch_size):
        # (num_layers*num_directions, batch_size, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers*hidden_dim)
        return h.transpose(0,1).contiguous().view(batch_size, -1)
    


    def _flatten_hidden(self, hidden, batch_size):
        if self.bidirectional or self.num_layers > 1:
            if self.rnn_type =='lstm':
                hidden = torch.cat(hidden[0].view(batch_size, self.hidden_size*self.hidden_factor),hidden[1].view(batch_size, self.hidden_size*self.hidden_factor), 1)
            else:
                hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            if self.rnn_type =='lstm':
                hidden = torch.cat((hidden[0].squeeze(),hidden[1].squeeze()))
            else:
                hidden = hidden.squeeze()   
        return hidden
        # if h is None:
        #     return None
        # elif isinstance(h, tuple): # LSTM
        #     X = torch.cat([self._flatten(h[0], batch_size), self._flatten(h[1], batch_size)], 1)
        # else: # GRU
        #     X = self._flatten(h, batch_size)
        # return X


class RNNDecoder(nn.Module):
    
    def __init__(self, device,vocab_size, embedding, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, dropout, max_sequence_length, num_layers=1, bidirectional=False,tie_embeddings=False ):
        super(RNNDecoder, self).__init__()
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise Exception("Unknown RNN type for encoder. Valid options: rnn, gru, lstm .")
        
        # Embedding layer
        self.embedding = embedding
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        
        
        self.rnn = rnn(input_size=self.embedding_size,
                       hidden_size=self.hidden_size,
                       num_layers= self.num_layers,
                       bidirectional=self.bidirectional,
                       dropout=self.dropout,
                       batch_first=True)
        
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
    
        # If set, tie weights of output layer to weights of embedding layer
        if self.tie_embeddings:
            # Map hidden_dim to embed_dim (can be dropped if hidden_dim=embed_dim)
            self.hidden_to_embed = nn.Linear(hidden_size * (2 if bidirectional else 1),  self.embedding_size)
            # Weight matrix of self.out has now the same dimension as embedding layer
            self.outputs2vocab = nn.Linear(self.embedding_size, vocab_size)
            # Set weights of output layer to embedding weights. Backprop seems to work quite fine with that.
            self.outputs2vocab.weight.data = self.embedding.embeddings.word_embeddings.weight
        else:
            self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, inputs, z, lengths=None):
        
        # Flatten params for data parallel
        self.rnn.flatten_parameters()
        
        batch_size, num_steps = inputs.shape

        hidden = self.latent2hidden(z)
        hidden = self._unflatten_hidden(hidden, batch_size)

        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(inputs.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(inputs.data - self.sos_idx) * (inputs.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = inputs.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
        else:
            decoder_input_sequence= inputs

        x  = self.embedding(decoder_input_sequence).last_hidden_state
        x = self.embedding_dropout(x)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths)
        
        outputs, _ = self.rnn(x, hidden)
        
        if lengths is not None:
            outputs = pad_packed_sequence(outputs)
        
        if self.tie_embeddings:
            outputs = self.outputs2vocab(self.hidden_to_embed(outputs.squeeze(dim=1)))
        else:
            #output = F.log_softmax(self.out(self.last_dropout(output.squeeze(dim=1))), dim=1)
            outputs = self.outputs2vocab(outputs.squeeze(dim=1))

        return outputs
        
        # "Expand" z vector
        #X = self.z_to_hidden(z)
        
        # Unflatten hidden state for GRU or LSTM
        # hidden = self._unflatten_hidden(X, batch_size)
        # hidden = self._unflatten_hidden(X, batch_size)
        # Restructure shape of hidden state to accommodate bidirectional encoder (decoder is unidirectional)
        # hidden = self._init_hidden_state(z)
        # # Create SOS token tensor as first input for decoder
        # # input = torch.LongTensor([[Token.SOS]] * batch_size).to(self.device)
        # input= inputs[:,0].unsqueeze(1)
        # # Decide whether to do teacher forcing or not
        # # use_teacher_forcing = random.random() < self.params.teacher_forcing_prob
        # use_teacher_forcing = self.training
        # # Initiliaze loss
        # loss = 0
        # outputs = torch.zeros((batch_size, num_steps), dtype=torch.long).to(self.device)
        # if use_teacher_forcing:
        #     x  = self.embedding(inputs).last_hidden_state
        #     if lengths is not None:
        #         x = pack_padded_sequence(x, lengths)
        #         # Push through RNN layer (the ouput is irrelevant)
        #         # Push through RNN layer (the ouput is irrelevant)
        #     outputs, _ = self.rnn(x, hidden)
        #     if lengths is not None:
        #         outputs = pad_packed_sequence(outputs)
        #         # Unpack the packed sequence
            # for i in range(num_steps):
            #     output, hidden = self._step(input, hidden)
            #     topv, topi = output.topk(1)
            #     outputs[:,i] = topi.detach().squeeze()
            #     #print(output[0], inputs[:, i][0])
            #     # loss += self.criterion(output, inputs[:, i])
            #     input = inputs[:, i].unsqueeze(dim=1)
        # else:
        #     for i in range(num_steps):
        #         output, hidden = self._step(input, hidden)
        #         log_output = F.log_softmax(output)
        #         topv, topi = log_output.topk(1)
        #         input = topi.detach()
        #         outputs[:, i] = topi.detach().squeeze()
        #         #print(topi[0], inputs[:, i][0])
        #         # loss += self.criterion(output, inputs[:, i])
        #         if input[0].item() == Token.EOS:
                    # break

        

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
        X = self.embedding(input).last_hidden_state
        # Push input word through rnn layer with current hidden state
        output, hidden = self.rnn(X, hidden)
        # Push output through linear layer to get to vocab_size

        if self.params.tie_embeddings == True:
            output = self.out(self.hidden_to_embed(output.squeeze(dim=1)))
        else:
            #output = F.log_softmax(self.out(self.last_dropout(output.squeeze(dim=1))), dim=1)
            output = self.out(output.squeeze(dim=1))
        # return the output (batch_size, vocab_size) and new hidden state
        return output, hidden

    def _unflatten_hidden(self, hidden, batch_size):
        if self.bidirectional or self.num_layers > 1:
            if self.rnn_type =='lstm':
                hidden_split = torch.split(hidden, int(hidden.shape[1]/2), dim=1)
                hidden = (hidden_split[0].view(self.hidden_factor, batch_size, self.hidden_size), hidden_split[1].view(self.hidden_factor, batch_size, self.hidden_size))
            else:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            if self.rnn_type =='lstm':
                hidden_split = torch.split(hidden, int(hidden.shape[1]/2), dim=1)
                hidden = (hidden_split[0].unsqueeze(0),hidden_split[1].unsqueeze(0)  )
            else:
                hidden = hidden.unsqueeze(0)  
        return hidden
        # if X is None:
        #     return None
        # elif self.params.rnn_type == RnnType.LSTM:  # LSTM
        #     X_split = torch.split(X, int(X.shape[1]/2), dim=1)
        #     h = (self._unflatten(X_split[0], batch_size), self._unflatten(X_split[1], batch_size))
        # else:  # GRU
        #     h = self._unflatten(X, batch_size)
        # return h

    def _unflatten(self, X, batch_size):
        # (batch_size, num_directions*num_layers*hidden_dim)    ==>
        # (batch_size, num_directions * num_layers, hidden_dim) ==>
        # (num_layers * num_directions, batch_size, hidden_dim) ==>
        return X.view(batch_size, self.params.num_layers * self.num_directions, self.params.rnn_hidden_dim).transpose(0, 1).contiguous()

