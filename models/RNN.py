import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class RNNEncoder(nn.Module):
    def __init__(self,vocab_size, embedding_size,hidden_size,latent_size,num_layers,dropout =0.1,bidirectional =True ,embedding_weights=None, hidden_mode='last',device='cuda'):
        super(RNNEncoder, self).__init__()
        
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers 
        self.bidirectional =bidirectional
        self.dropout = dropout
        self.embedding_weights = embedding_weights
        self.hidden_mode = hidden_mode

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = nn.GRU(input_size =self.embedding_size,hidden_size= self.hidden_size,num_layers = self.num_layers,batch_first=True,dropout= self.dropout, bidirectional =self.bidirectional)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.fc_mu = nn.Linear(in_features=self.hidden_size * self.hidden_factor, out_features=self.latent_size)
        self.fc_logvar = nn.Linear(in_features=self.hidden_size*self.hidden_factor, out_features=self.latent_size)

        self._init_weights(self.embedding_weights)
            
    def _init_weights(self,emb_weights):
      for m in self.modules():
        if isinstance(m, nn.Embedding):
           if not (emb_weights is None) :
             m.weight = emb_weights
           else:
            nn.init.uniform_(m.weight, -0.001, 0.001)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
         
    
    def forward(self, input_ids, input_ids_length):
        batch_size, seq_len = input_ids.shape

        input_embedding = self.embedding(input_ids)
        
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, input_ids_length.data.tolist(), batch_first=True, enforce_sorted=False)
        _ , hidden = self.rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        x_mu = self.fc_mu(hidden)
        x_logvar = self.fc_logvar(hidden)
        return x_mu, x_logvar

class RNNDecoder(nn.Module):
    def __init__(self,vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,dropout =0.1,bidirectional =False ,embedding_weights=None,tie_embedding=False,word_dropout_rate=0.0, device='cuda'):
        super(RNNDecoder, self).__init__()
        
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers 
        self.bidirectional =bidirectional
        self.dropout =dropout
        self.embedding_weights = embedding_weights
        self.tie_embedding= tie_embedding
        self.word_dropout_rate = word_dropout_rate
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id 
        self.unk_token_id = unk_token_id

        
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size * self.hidden_factor)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = nn.GRU(input_size =self.embedding_size,hidden_size= self.hidden_size,num_layers = self.num_layers,batch_first=True,dropout= self.dropout, bidirectional=self.bidirectional)
        self.outputs2vocab = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.vocab_size)
        if self.tie_embedding:
           self.hidden2emb = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.embedding_size)
           self.outputs2vocab = nn.Linear(self.embedding_size, self.vocab_size)
        
        self._init_weights(self.embedding_weights)
           
        
    def _init_weights(self,emb_weights):
      for m in self.modules():
        if isinstance(m, nn.Embedding):
           if not (emb_weights is None) :
             m.weight = emb_weights
           else:
            nn.init.uniform_(m.weight, -0.001, 0.001)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
      if self.tie_embedding: 
         self.outputs2vocab.weight = emb_weights
           
    
    

    def forward(self, z, dec_input_ids, dec_input_ids_length=None,skip_hidden_tr=False):
        batch_size, seq_len=  dec_input_ids.shape
        
        if self.word_dropout_rate > 0:
          # randomly replace decoder input with <unk>
          prob = torch.rand(dec_input_ids.size()).to(self.device)
          prob[(dec_input_ids.data - self.eos_token_id) * (dec_input_ids.data - self.pad_token_id) == 0] = 1
          dec_input_ids[prob < self.word_dropout_rate] = self.unk_token_id


        input_embedding = self.embedding(dec_input_ids)   
        if not (dec_input_ids_length is None):
          packed_input = rnn_utils.pack_padded_sequence(input_embedding, dec_input_ids_length.data.tolist(), batch_first=True, enforce_sorted=False)
        else:
          packed_input = input_embedding
      
        if skip_hidden_tr:
            hidden = z
        else:
            hidden = self.latent2hidden(z)
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
          
        outputs , hidden = self.rnn(packed_input, hidden)
        
        if not (dec_input_ids_length is None):
          padded_outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        else:
          padded_outputs= outputs

        padded_outputs = padded_outputs.contiguous()
        if self.tie_embedding: 
          x= self.hidden2emb(padded_outputs)
          x= self.outputs2vocab(x)

        else:
          x= self.outputs2vocab(padded_outputs)

        return x, hidden
    
    def inference(self, n=4, max_sequence_length=100, z=None,temp=1.0, mode='sample_temp'):
        if z is None:
          batch_size = n
          z = torch.randn([batch_size, self.latent_size])
        else:
          batch_size = z.size(0)
        
        z = torch.tensor(z, device = self.device)

        generations = torch.full(
                (batch_size, max_sequence_length+1, ),
                self.pad_token_id,
                dtype=torch.long,
                device= self.device,
            )
        
        generations[:,0] = self.bos_token_id
        num_gen_tokens = 0
        unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device= self.device)
        skip_latent_transformation= False
        hiddens_cache = torch.zeros((self.hidden_factor, batch_size, self.hidden_size), device= self.device)
        while True:
          # B* x S x F
          if num_gen_tokens == 0:
            unfinished_latents = z[unfinished_mask]
          else:
            unfinished_latents = hiddens_cache[:,unfinished_mask]

          # B* x T
          unfinished_generations = generations[unfinished_mask, : num_gen_tokens + 1]

          # B* x T x K
          logits, hiddens = self.forward(unfinished_latents,unfinished_generations,skip_hidden_tr=skip_latent_transformation)
          hiddens_cache[:,unfinished_mask,:] = hiddens
          
          if num_gen_tokens == 0:
            skip_latent_transformation = True
          # B* x K
          logits = logits[:, -1, :]
          probs = torch.nn.functional.softmax(logits/temp, dim=-1)
          # B*
          if mode == 'greedy':
            gen_token_ids = probs.argmax(dim=-1)
          else: 
            gen_token_ids = torch.multinomial(probs, 1).squeeze(1)
            
          # B*
          generations[unfinished_mask, num_gen_tokens+1] = gen_token_ids
          # B*
          unfinished_mask[unfinished_mask == True] = ( gen_token_ids != self.eos_token_id )
          num_gen_tokens += 1
          if (not unfinished_mask.any()) or (num_gen_tokens >= max_sequence_length):
              break
            # B x T
          
        generations = generations[:, 1 : num_gen_tokens + 1]
        return generations, z