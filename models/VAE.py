import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RNN import RNNEncoder,RNNDecoder
from models.transformer import Transformer_Encoder,Transformer_Decoder, AttentionTransformerDecoder

import logging

logger = logging.getLogger(__name__)

class RNNVariationalAutoencoder(nn.Module):

    def __init__(self,vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,dropout,embedding_weights=None,tie_embedding=False, word_dropout_rate=0.0, hidden_mode = 'last', device='cuda'):
        super(RNNVariationalAutoencoder, self).__init__()
        self.encoder = RNNEncoder(vocab_size, embedding_size,hidden_size,latent_size,num_layers,dropout,True ,embedding_weights, hidden_mode, device)
        self.decoder = RNNDecoder(vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,dropout,False ,embedding_weights,tie_embedding,word_dropout_rate, device)

    def forward(self,input_ids, input_ids_length, dec_input_ids, dec_input_ids_length):
        latent_mu, latent_logvar = self.encoder(input_ids,input_ids_length)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon,_ = self.decoder(latent,dec_input_ids,dec_input_ids_length)
        return x_recon, latent_mu, latent_logvar
    
    def generate(self,input_ids, input_ids_length,max_sequence_length=100,temp=0.8, mode='sample_temp'):
        latent_mu, latent_logvar = self.encoder(input_ids,input_ids_length)
        latent = self.latent_sample(latent_mu, latent_logvar)
        hyp,_ = self.decoder.inference(max_sequence_length=max_sequence_length, z=latent,temp=temp, mode=mode)
        return hyp
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

class TransformerVariationalAutoencoder(nn.Module):

    def __init__(self,vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout=0.1,embedding_weights=None,tie_embedding=False, word_dropout_rate=0.0,  hidden_mode='mean',device='cuda'):
        super(TransformerVariationalAutoencoder, self).__init__()

        self.encoder = Transformer_Encoder(vocab_size, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout,embedding_weights,hidden_mode,device)
        self.decoder = Transformer_Decoder(vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout ,embedding_weights,tie_embedding,word_dropout_rate, device)
    
    
    def forward(self,input_ids, dec_input_ids, src_pad_mask, target_pad_mask, target_lookahead_mask):
        latent_mu, latent_logvar = self.encoder(input_ids,src_pad_mask)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent,dec_input_ids, target_pad_mask, target_lookahead_mask)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    def generate(self,input_ids, src_pad_mask ,max_sequence_length=100,temp=0.8,mode='sample_temp' ):
        latent_mu, latent_logvar = self.encoder(input_ids,src_pad_mask)
        latent = self.latent_sample(latent_mu, latent_logvar)
        hyp,_ = self.decoder.inference(max_sequence_length=max_sequence_length, z=latent,temp=temp,mode=mode)
        return hyp

class AttentionTransformerVariationalAutoencoder(nn.Module):
    def __init__(self,vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout =0.1,embedding_weights=None,tie_embedding=False, word_dropout_rate=0.0,device='cuda'):
        super(AttentionTransformerVariationalAutoencoder, self).__init__()
        self.encoder = Transformer_Encoder(vocab_size, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout,embedding_weights,'all',device)
        self.decoder = AttentionTransformerDecoder(vocab_size,bos_token_id,eos_token_id,pad_token_id, unk_token_id, embedding_size,hidden_size,latent_size,num_layers,nhead,dropout ,embedding_weights,tie_embedding,word_dropout_rate, device)
   
    def forward(self,input_ids, dec_input_ids, src_pad_mask, target_pad_mask, target_lookahead_mask):
        latent_mu, latent_logvar = self.encoder(input_ids,src_pad_mask)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent,dec_input_ids,src_pad_mask, target_pad_mask, target_lookahead_mask)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    def generate(self,input_ids, src_pad_mask,max_sequence_length=100,temp=0.8, mode='sample_temp'):
        latent_mu, latent_logvar = self.encoder(input_ids,src_pad_mask)
        latent = self.latent_sample(latent_mu, latent_logvar)
        hyp,_ = self.decoder.inference(max_sequence_length=max_sequence_length, memory_key_padding_mask=src_pad_mask, z=latent,temp=temp, mode = mode)
        return hyp



if __name__ == "__main__":
    
    from data_utils.tokenizer import get_tokenizer
    from data_utils.librispeech_dataset import LibriSpeechDataset
    params={'src': 'gpt2', 'cache_dir': './cache'}
    tokenizer, embedding = get_tokenizer('gpt',**params)
    word_embeddings = embedding.transformer.wte.weight  # Word Token Embeddings 
    train_set = LibriSpeechDataset(csv_file="./data_dir/train-clean-100.csv", root_dir='./',tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, collate_fn= train_set.collate_fn ,batch_size=32, shuffle=True )
    batch = next(iter(train_loader))
    device= 'cpu'




    
    input_ids, input_ids_lens = batch['input_ids']
    dec_input_ids, dec_input_ids_lens = batch['dec_input_ids']
    input_ids,dec_input_ids= input_ids.to(device),dec_input_ids.to(device)

    # args={'vocab_size':len(tokenizer), 
    #       'bos_token_id': tokenizer.bos_token_id ,
    #       'eos_token_id': tokenizer.eos_token_id,
    #       'pad_token_id': tokenizer.pad_token_id, 
    #       'unk_token_id': tokenizer.unk_token_id,
    #       'embedding_size':768,
    #       'hidden_size': 256, 
    #       'dropout':0.1,
    #       'latent_size':16,
    #       'num_layers':1, 
    #       'embedding_weights': word_embeddings,
    #       'tie_embedding': True,
    #       'word_dropout_rate' : 0.5,
    #       'hidden_mode' : 'mean',
    #       'device' : device

    # }
    # vae = RNNVariationalAutoencoder(**args).to(device)
    # out, mu, var = vae(input_ids,input_ids_lens,dec_input_ids,dec_input_ids_lens)
    # print("RNNVAE test successfully!")


    # args={'vocab_size':len(tokenizer), 
    #       'bos_token_id': tokenizer.bos_token_id ,
    #       'eos_token_id': tokenizer.eos_token_id,
    #       'pad_token_id': tokenizer.pad_token_id, 
    #       'unk_token_id': tokenizer.unk_token_id,
    #       'embedding_size':768,
    #       'nhead': 4, 
    #       'dropout':0.1,
    #       'hidden_size': 768, 
    #       'latent_size':16,
    #       'num_layers':1, 
    #       'embedding_weights': word_embeddings,
    #       'tie_embedding': True,
    #       'word_dropout_rate' : 0.5,
    #       'hidden_mode' : 'mean',
    #       'device' : device

    # }
    
    # from transformer import generate_pad_mask,generate_square_subsequent_mask
    # trg_lookahead_mask= generate_square_subsequent_mask(dec_input_ids.shape[1]).to(device)
    # trg_pad_mask= generate_pad_mask(dec_input_ids, tokenizer.pad_token_id)
    # src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)

    # vae = TransformerVariationalAutoencoder(**args).to(device)
    # out, mu, var = vae(input_ids,dec_input_ids,src_pad_mask,trg_pad_mask,trg_lookahead_mask )
    # print("TransformerVAE test successfully!")


    args={'vocab_size':len(tokenizer), 
          'bos_token_id': tokenizer.bos_token_id ,
          'eos_token_id': tokenizer.eos_token_id,
          'pad_token_id': tokenizer.pad_token_id, 
          'unk_token_id': tokenizer.unk_token_id,
          'embedding_size':768,
          'nhead': 4, 
          'dropout':0.1,
           'hidden_size': 768, 
          'latent_size':16,
          'num_layers':1, 
          'embedding_weights': word_embeddings,
          'tie_embedding': True,
          'word_dropout_rate' : 0.5,
          'device' : device

    }
    
    from transformer import generate_pad_mask,generate_square_subsequent_mask
    trg_lookahead_mask= generate_square_subsequent_mask(dec_input_ids.shape[1]).to(device)
    trg_pad_mask= generate_pad_mask(dec_input_ids, tokenizer.pad_token_id)
    src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)

    vae = AttentionTransformerVariationalAutoencoder(**args).to(device)
    out, mu, var = vae(input_ids,dec_input_ids,src_pad_mask,trg_pad_mask,trg_lookahead_mask )
    print("AttentionTransformerVAE test successfully!")
