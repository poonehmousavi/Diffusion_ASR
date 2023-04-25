import torch
from torch import  Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer
import math


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def generate_pad_mask(padded_input: Tensor, pad_token_id) -> Tensor:
    """Generates ask for padde tokens``."""
    return padded_input.eq(pad_token_id).to(padded_input.device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)].permute(1,0,2)
        return self.dropout(x)


class Transformer_Encoder(nn.Module):
    def __init__(self,vocab_size, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout =0.1,embedding_weights=None, hidden_mode='all',device='cuda'):
        super(Transformer_Encoder, self).__init__()
        
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.dropout =dropout
        self.embedding_weights= embedding_weights
        self.hidden_mode = hidden_mode

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_encoder = PositionalEncoding(self.embedding_size, self.dropout)
        encoder_layers = TransformerEncoderLayer(self.embedding_size, nhead, self.hidden_size,dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc_mu = nn.Linear(in_features=self.embedding_size , out_features=latent_size)
        self.fc_logvar = nn.Linear(in_features=self.embedding_size, out_features=latent_size)

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
         
    
    def forward(self, input_ids,src_key_padding_mask=None):

        input_embedding = self.embedding(input_ids)
        src = self.pos_encoder(input_embedding)

        outputs = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        x_mu = self.fc_mu(outputs)
        x_logvar = self.fc_logvar(outputs)

        if self.hidden_mode == "sum":
           x_mu,x_logvar = torch.sum(x_mu,dim=1),  torch.mean(x_logvar,dim=1)
        elif self.hidden_mode == "last":
           x_logvar = x_mu[:,-1,:],  x_logvar[:,-1,:]
        elif self.hidden_mode == "mean":
          x_mu,x_logvar = torch.mean(x_mu,dim=1),  torch.mean(x_logvar,dim=1)

        # if hidden_mode is not passed, it returns all the latent for all steps
           
        return  x_mu,x_logvar

class Transformer_Decoder(nn.Module):
    def __init__(self,vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout =0.1 ,embedding_weights=None,tie_embedding=False,word_dropout_rate=0.0, device='cuda'):
        super(Transformer_Decoder, self).__init__()
        
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers 
        self.dropout =dropout
        self.embedding_weights = embedding_weights
        self.tie_embedding= tie_embedding
        self.word_dropout_rate = word_dropout_rate
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id 
        self.unk_token_id = unk_token_id
        

        self.latent_to_embedding = nn.Linear(self.latent_size, self.embedding_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_encoder = PositionalEncoding(self.embedding_size, self.dropout)
        encoder_layers = TransformerEncoderLayer(self.embedding_size, nhead, self.hidden_size, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
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
  
    
    def forward(self, z, dec_input_ids,src_key_padding_mask=None, look_ahead_mask=None):
        if self.word_dropout_rate > 0:
          # randomly replace decoder input with <unk>
          prob = torch.rand(dec_input_ids.size()).to(self.device)
          prob[(dec_input_ids.data - self.eos_token_id) * (dec_input_ids.data - self.pad_token_id) == 0] = 1
          dec_input_ids[prob < self.word_dropout_rate] = self.unk_token_id

        input_embedding = self.embedding(dec_input_ids)  
        src = self.pos_encoder(input_embedding) 
        latent = self.latent_to_embedding(z)
        src = src + latent.unsqueeze(1)
        outputs = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask,  mask=look_ahead_mask)
        x= self.outputs2vocab(outputs)
        return x
    
    def inference(self, n=4, max_sequence_length=100, z=None, temp=1.0, mode='sample_temp'):
        if z is None:
          batch_size = n
          z = torch.randn([batch_size, self.latent_size])
        else:
          batch_size = z.size(0)
        
        z = torch.tensor(z, device=self.device)

        generations = torch.full(
                (batch_size, max_sequence_length+1, ),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
        
        generations[:,0] = self.bos_token_id
        num_gen_tokens = 0
        unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        while True:
          # B* x S x F
          unfinished_latents = z[unfinished_mask]
          # B* x T
          unfinished_generations = generations[unfinished_mask, : num_gen_tokens + 1]
          trg_lookahead_mask= generate_square_subsequent_mask(num_gen_tokens + 1).to(self.device)

          # B* x T x K
          logits = self.forward(unfinished_latents,unfinished_generations, look_ahead_mask=trg_lookahead_mask)
          
          if num_gen_tokens == 0:
            skip_latent_transformation = True
          # B* x K
          logits = logits[:, -1, :]
          probs = torch.nn.functional.softmax(logits/temp, dim=-1)
          # print(probs.shape)
          # B*
          if mode == 'greedy':
            gen_token_ids = probs.argmax(dim=-1)
          else: 
            # probs = torch.nn.functional.softmax(logits/temp, dim=-1)
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

class AttentionTransformerDecoder(nn.Module):
    def __init__(self,vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout =0.1 ,embedding_weights=None,tie_embedding=False,word_dropout_rate=0.0, device='cuda'):
        super(AttentionTransformerDecoder, self).__init__()
        
        self.device = device
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers 
        self.dropout = dropout
        self.embedding_weights = embedding_weights
        self.tie_embedding= tie_embedding
        self.word_dropout_rate = word_dropout_rate
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id 
        self.unk_token_id = unk_token_id
        
    
        self.latent_to_embedding = nn.Linear(self.latent_size, self.embedding_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_encoder = PositionalEncoding(self.embedding_size, self.dropout)
        decoder_layers = TransformerDecoderLayer(self.embedding_size, nhead, self.hidden_size, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        self.outputs2vocab = nn.Linear(self.embedding_size, self.vocab_size)
        self.tanh = nn.Tanh()

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
  
    
    def forward(self, z, dec_input_ids,memory_key_padding_mask,tgt_key_padding_mask=None, look_ahead_mask=None):
        
        if self.word_dropout_rate > 0:
          # randomly replace decoder input with <unk>
          prob = torch.rand(dec_input_ids.size()).to(self.device)
          prob[(dec_input_ids.data - self.eos_token_id) * (dec_input_ids.data - self.pad_token_id) == 0] = 1
          dec_input_ids[prob < self.word_dropout_rate] = self.unk_token_id        


        input_embedding = self.embedding(dec_input_ids)  
        src = self.pos_encoder(input_embedding) 
        memory= self.latent_to_embedding(z)
        outputs = self.transformer_decoder(src,memory,memory_key_padding_mask =memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,  tgt_mask =look_ahead_mask)
        x= self.outputs2vocab(outputs)

        return x

    def inference(self, n=4, sample_seq_len=20, max_sequence_length=100, memory_key_padding_mask=None, z=None,temp=1.0, mode='sample_temp'):
        if z is None:
          batch_size = n
          z = torch.randn([batch_size, sample_seq_len, self.latent_dims])
        else:
          batch_size = z.size(0)
        
        z = torch.tensor(z, device=self.device)

        generations = torch.full(
                (batch_size, max_sequence_length+1, ),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
        
        generations[:,0] = self.bos_token_id
        num_gen_tokens = 0
        unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        while True:
          # B* x S x F
          unfinished_latents = z[unfinished_mask]
          # B* x T
          unfinished_generations = generations[unfinished_mask, : num_gen_tokens + 1]
          trg_lookahead_mask= generate_square_subsequent_mask(num_gen_tokens + 1).to(self.device)
          if not (memory_key_padding_mask is None):
            unfinished_src_padded_mask=memory_key_padding_mask[unfinished_mask]
          else:
            unfinished_src_padded_mask =None


          # B* x T x K
          logits = self.forward(unfinished_latents,unfinished_generations, memory_key_padding_mask=unfinished_src_padded_mask,  look_ahead_mask=trg_lookahead_mask)
          
          if num_gen_tokens == 0:
            skip_latent_transformation = True
          # B* x K
          logits = logits[:, -1, :]
          probs = torch.nn.functional.softmax(logits/temp, dim=-1)
          # print(probs.shape)
          # B*
          if mode == 'greedy':
            gen_token_ids = probs.argmax(dim=-1)
          else: 
            # probs = torch.nn.functional.softmax(logits/temp, dim=-1)
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
     

