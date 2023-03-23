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


