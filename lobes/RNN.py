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
                dropout, num_layers=1, bidirectional=False,freeze_embedding=True ):
        super(RNNEncoder, self).__init__()
        self.device = device


        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.freeze_embedding= freeze_embedding

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise Exception("Unknown RNN type for encoder. Valid options: rnn, gru, lstm .")
        
        # Embedding layer
        # self.embedding = embedding
        # if self.freeze_embedding:
        #     self.embedding.train()  # we keep it to train to have dropout and LN computed adequaly
        #     for param in self.embedding.parameters():
        #         param.requires_grad = False

        self.embedding = embedding
        # self.embedding.weight = embedding.transformer.wte.weight 
        if self.freeze_embedding:
            self.embedding.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.embedding.parameters():
                param.requires_grad = False
                
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
    
    
    def forward(self, inputs,lengths=None):

        # Flatten params for data parallel
        self.rnn.flatten_parameters()

        # inputs.shape = (batch_size, seq_len)
        batch_size, seq_len = inputs.shape
        # Push through embedding layer ==> X.shape = (batch_size, seq_len, embed_dim)
        with torch.set_grad_enabled(not self.freeze_embedding):
            # x  = self.embedding(inputs).last_hidden_state
            x  = self.embedding(inputs)
        
        
        # x =F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = self.l5(x)

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



class RNNDecoder(nn.Module):
    
    def __init__(self, device,vocab_size, embedding, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, dropout, max_sequence_length, num_layers=1, bidirectional=False,tie_embeddings=False,freeze_embedding=True ):
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
        self.freeze_embedding= freeze_embedding

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise Exception("Unknown RNN type for encoder. Valid options: rnn, gru, lstm .")
        
        # Embedding layer
        # self.embedding = embedding
        # if self.freeze_embedding:
        #     self.embedding.train()  # we keep it to train to have dropout and LN computed adequaly
        #     for param in self.embedding.parameters():
        #         param.requires_grad = False
        self.embedding = embedding
        # self.embedding.weight = embedding.transformer.wte.weight 
        if self.freeze_embedding:
            self.embedding.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.embedding.parameters():
                param.requires_grad = False



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
        # self.l2 = nn.Linear(hidden_size * self.hidden_factor, 128)
        # self.l3 = nn.Linear(128, 256)
        # self.l4 = nn.Linear(256, 512)
        # self.l5 = nn.Linear(512, self.embedding_size)
    
        # If set, tie weights of output layer to weights of embedding layer
        if self.tie_embeddings:
            # Map hidden_dim to embed_dim (can be dropped if hidden_dim=embed_dim)
            self.hidden_to_embed = nn.Linear(hidden_size * (2 if bidirectional else 1),  self.embedding_size)
            # Weight matrix of self.out has now the same dimension as embedding layer
            self.outputs2vocab = nn.Linear(self.embedding_size, vocab_size)
            # Set weights of output layer to embedding weights. Backprop seems to work quite fine with that.
            self.outputs2vocab.weight = self.embedding.weight
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
        with torch.set_grad_enabled(not self.freeze_embedding):
            # x  = self.embedding(decoder_input_sequence).last_hidden_state
            x  = self.embedding(decoder_input_sequence)
        x = self.embedding_dropout(x)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths)
        
        outputs, _ = self.rnn(x, hidden)
        
        if lengths is not None:
            outputs = pad_packed_sequence(outputs)
        # outputs =F.relu(self.l2(outputs))
        # outputs = F.relu(self.l3(outputs))
        # outputs = F.relu(self.l4(outputs))
        # outputs = self.l5(outputs)
        
        if self.tie_embeddings:
            logits = self.outputs2vocab(self.hidden_to_embed(outputs.squeeze(dim=1)))
        else:
            #output = F.log_softmax(self.out(self.last_dropout(output.squeeze(dim=1))), dim=1)
            logits = self.outputs2vocab(outputs.squeeze(dim=1))

        return logits
    

    @torch.no_grad()
    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = (torch.randn([batch_size, self.latent_size])).to(self.device)
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        hidden = self._unflatten_hidden(hidden, batch_size)

        # if self.bidirectional or self.num_layers > 1:
        #     # unflatten hidden state
        #     hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        # hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=torch.LongTensor()).to(self.device)  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=torch.LongTensor()).to(self.device)
        sequence_mask = torch.ones(batch_size, out=torch.BoolTensor()).to(self.device)
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=torch.LongTensor()).to(self.device)

        generations = torch.LongTensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).to(self.device)

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = (torch.Tensor(batch_size).fill_(self.sos_idx).long()).to(self.device)

            input_sequence = input_sequence.unsqueeze(1)

            # input_embedding = self.embedding(input_sequence).last_hidden_state
            input_embedding = self.embedding(input_sequence)
            output, hidden = self.rnn(input_embedding, hidden)

            # output = F.relu(self.l2(output))
            # output =F.relu(self.l3(output))
            # output = F.relu(self.l4(output))
            # output = self.l5(output)

            if self.tie_embeddings:
                logits = self.outputs2vocab(self.hidden_to_embed(output.squeeze(dim=1)))
            else:
                 logits = self.outputs2vocab(output.squeeze(dim=1))
            
            prob = F.log_softmax(logits, dim=1)

            input_sequence = self._sample(prob)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=torch.LongTensor()).to(self.device)

            t += 1

        return generations, z
    
    @torch.no_grad()
    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample
    
    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

    def _init_hidden_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        elif isinstance(encoder_hidden, tuple): # LSTM
            return tuple([self._concat_directions(h) for h in encoder_hidden])
        else: # GRU
            return self._concat_directions(encoder_hidden)


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
                hidden = (hidden_split[0].unsqueeze(0),hidden_split[1].unsqueeze(0))
            else:
                hidden = hidden.unsqueeze(0)
        return hidden


    def _unflatten(self, X, batch_size):
        return X.view(batch_size, self.params.num_layers * self.num_directions, self.params.rnn_hidden_dim).transpose(0, 1).contiguous()

