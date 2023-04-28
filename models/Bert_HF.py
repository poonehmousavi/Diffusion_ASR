import torch 
from torch import Tensor
import torch.nn as nn
from transformers import BertForMaskedLM,BertConfig,BertModel
import logging

logger = logging.getLogger(__name__)

class HuggingFaceBERT_LM(nn.Module):
    def __init__(self, src, cache_dir, freeze=True, device='cuda'):
        super(HuggingFaceBERT_LM, self).__init__()
        config = BertConfig.from_pretrained(src)
        config.is_decoder = True
        self.model= BertForMaskedLM.from_pretrained(src, cache_dir=cache_dir)
        self.embedding_size=768

        self.device = device
        self.freeze = freeze
        if self.freeze:
            logger.warning(
                "HuggingFaceBERT_LM  is frozen."
            )
            self.model.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, input_embedding: Tensor, attention_mask: Tensor =None):
        with torch.set_grad_enabled(not self.freeze):
            token_type_ids = torch.zeros(input_embedding.shape[0],input_embedding.shape[1], dtype=torch.long).to(input_embedding.device)
            output = self.model(inputs_embeds=input_embedding, attention_mask=attention_mask,token_type_ids=token_type_ids).logits
        return output
    
    def inference(self, n=4, sample_seq_len=20, max_sequence_length=100,padding_mask=None, z=None,temp=1.0, mode='sample_temp'):
        if z is None:
          batch_size = n
          z = torch.randn([batch_size, sample_seq_len, self.embedding_size])
        else:
          batch_size = z.size(0)
          sample_seq_len = z.size(1)
        
        z = torch.tensor(z, device=self.device)
        
        logits = self.forward(input_embedding=z,attention_mask=padding_mask)
        probs = torch.nn.functional.softmax(logits/temp, dim=-1)
        if mode == 'greedy':
            generations = probs.argmax(dim=-1)
        else:  
            generations = torch.multinomial(probs.view(-1, logits.shape[-1]), 1).view(batch_size,sample_seq_len,-1).squeeze(-1)

        return generations
            
        




class HuggingFaceBERT_Encoder(nn.Module):
    def __init__(self, src, cache_dir,mask_token_id, pad_token_id,masking_ratios=0.0, freeze=True, device='cuda'):
        super(HuggingFaceBERT_Encoder, self).__init__()
        config = BertConfig.from_pretrained(src)
        config.is_decoder = False
        self.model= BertForMaskedLM.from_pretrained(src,config=config, cache_dir=cache_dir)
        self.device = device
        self.freeze = freeze
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.masking_ratios = masking_ratios
        if self.freeze:
            logger.warning(
                "HuggingFaceGPT_Encoder  is frozen."
            )
            self.model.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, input_ids: Tensor, attention_mask: Tensor =None, retun_hidden_layer=0):
        with torch.set_grad_enabled(not self.freeze):
          if self.masking_ratios > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_ids.size()).to(self.device)
            prob[(input_ids.data) * (input_ids.data - self.pad_token_id) == 0] = 1
            input_ids[prob < self.masking_ratios] = self.mask_token_id    
        token_type_ids = torch.zeros_like(input_ids).to(input_ids.device)
        output = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask, output_hidden_states=True).hidden_states[retun_hidden_layer]
        # output = self.model(**input_ids,  output_hidden_states=True).hidden_states[retun_hidden_layer]

        return output
    

class HuggingFaceBERT_AE(nn.Module):
    def __init__(self, src, cache_dir,mask_token_id, pad_token_id ,masking_ratios=0.0,freeze_encoder=True,freeze_decoder=True, device='cuda'):
        super(HuggingFaceBERT_AE, self).__init__()
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.encoder = HuggingFaceBERT_Encoder(src, cache_dir,mask_token_id,pad_token_id,masking_ratios, freeze_encoder, device)
        self.decoder= HuggingFaceBERT_LM(src, cache_dir, freeze_decoder,device)
    
    def forward(self,iput, mask,retun_hidden_layer=3):
        embedding= self.encoder(iput,mask,retun_hidden_layer)
        output= self.decoder(embedding)
        return output
if __name__ == "__main__": 

    from data_utils.tokenizer import get_tokenizer
    from data_utils.librispeech_dataset import LibriSpeechDataset
    params={'src': 'bert-base-uncased', 'cache_dir': './cache'}
    tokenizer, embedding = get_tokenizer('bert',**params)
    train_set = LibriSpeechDataset(csv_file="./data_dir/train-clean-100.csv", root_dir='./',tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, collate_fn= train_set.collate_fn ,batch_size=32, shuffle=True )
    batch = next(iter(train_loader))
    device= 'cpu'


    input_ids, input_ids_lens = batch['input_ids']
    dec_input_ids, dec_input_ids_lens = batch['dec_input_ids']
    input_ids,dec_input_ids= input_ids.to(device),dec_input_ids.to(device)

    args={
          'pad_token_id': tokenizer.pad_token_id, 
          'mask_token_id': tokenizer.mask_token_id,
          'src': 'bert-base-uncased',
          'cache_dir': 'cache_dir/',
          'masking_ratios': 0.0,
          'freeze_encoder': True,
          'freeze_decoder': True,
          'device' : device

    }


    
    from transformer import generate_pad_mask,generate_square_subsequent_mask
    src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)

    ae = HuggingFaceBERT_AE(**args).to(device)
    # output = ae(input_ids,src_pad_mask)
    # input_ids = tokenizer("she paused and he came in not lifting his eyes to hers always when he crossed that threshold he had come with his head up and his wistful gaze seeking hers", return_tensors="pt")['input_ids']
    # src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
    
    latent = ae.encoder(input_ids,src_pad_mask,retun_hidden_layer=0)

    hyp = ae.decoder.inference(z=latent,padding_mask=None,mode='greedy')
    texts= tokenizer.batch_decode(hyp)

    hyp = ae.decoder.inference(mode='temp_sample')
    print(texts[:1])
    # print(tokenizer.batch_decode(input_ids[:1]))
    print("bert ae test successfully!")
