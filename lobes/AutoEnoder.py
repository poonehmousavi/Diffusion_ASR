import torch
import torch.nn as nn

class RNN_AE(nn.Module):
    def __init__(self, encoder, decoder,):
        super(RNN_AE,self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x, decoder_inputs,tokens_lens, tokens_bos_lens):
        _,_,z= self.encoder(x,tokens_lens)
        return self.decoder(decoder_inputs,z,tokens_bos_lens)
    
    def generate(self,x,tokens_lens):
        _,_,z= self.encoder(x,tokens_lens)
        hyp,z = self.decoder.inference(z=z)
        return hyp
    
    def inference(self,number_sample):
        hyp,z = self.decoder.inference(n=number_sample)
        return hyp

class RNN_VAE(nn.Module):
    def __init__(self, encoder, decoder,):
        super(RNN_VAE,self).__init__()

        self.encoder = encoder
        self.decoder = decoder
    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self,x, decoder_inputs,tokens_lens, tokens_bos_lens):
        _,_,z= self.encoder(x,tokens_lens)
        mu_logvar= z.view(-1, 2, self.encoder.latent_size//2)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(decoder_inputs,z,tokens_bos_lens), mu, logvar,z
    
    def generate(self,x,tokens_lens):
        _,_,z= self.encoder(x,tokens_lens)
        mu_logvar= z.view(-1, 2, self.encoder.latent_size//2)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        hyp,z = self.decoder.inference(z=z)
        return hyp
    
    def inference(self,number_sample):

        hyp,z = self.decoder.inference(n=number_sample)
        return hyp
    




