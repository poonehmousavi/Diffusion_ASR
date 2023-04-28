import pandas as pd
import torch
from transformers import BertModel, BertTokenizer,GPT2Tokenizer,GPT2Model,GPT2LMHeadModel,AutoTokenizer
import logging 


logger = logging.getLogger(__name__)


class char_tokenizer:
  def __init__(self,corpus_filepath, special_tokens):
    
    self.data = self.load_corpus(corpus_filepath)
    self.bos_token='@'
    self.eos_token='$'
    self.pad_token='#'
    self.unk_token='&'

    self.bos_token_id=0
    self.eos_token_id=1
    self.pad_token_id=2
    self.unk_token_id=3

    self.char_set=list(sorted(set("".join(self.data))))
    self.char_set.insert(self.bos_token_id,self.bos_token)
    self.char_set.insert(self.eos_token_id,self.eos_token)
    self.char_set.insert(self.pad_token_id,self.pad_token)
    self.char_set.insert(self.pad_token_id,self.pad_token)

    self.vocab_size= len(self.char_set)
  
  def load_corpus(self,path):
    data = pd.read_csv(path)
    train_data_txt = data['wrd']
    return train_data_txt
    

  def encode(self,data,add_special_tokens=False):
    """This function converts all the strings available in the data list into 
    a tensor of indexes. The conversion between char and index is done based on
    the content of the char_set. It also add the special token '@' to indicate
    begin-of-sentence. 
      
      Arguments
      ---------
      data : List
        A list containing strings to convert.
      
      Returns
      ---------
      data_index: torch.Tensor
        Tensor (N,L) containing the indexes corresponding to the input chars. N is 
        the number of text chunks and L is the number of char in each chunk.   
    """
    data_index = []
    for chunk in data:
      tensor = torch.zeros(len(chunk)).long()
      for i in range(len(chunk)):
        tensor[i] = self.char_set.index(chunk[i])
      data_index.append(tensor)
    # data_index = torch.stack(data_index)
    return data_index

  def decode(self,data,add_special_tokens=False):

    data_index = []
    for chunk in data:
      decoded_str=""
      for i in range(len(chunk)):
        if chunk[i] == self.eos_token_id:
          break
        decoded_str += self.char_set[chunk[i]]
      data_index.append(decoded_str)
    return data_index


def get_tokenizer(tokenizer_opt, **params):
    tokenizer, embedding_model = None, None
    
    if tokenizer_opt == 'char':
      bos = '@'
      eos = '$'
      pad = '#'
      unk = '&'
      special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad,'unk_token':unk}
      tokenizer= char_tokenizer("./sample_data/train-clean-100.csv",special_tokens_dict)
    
    elif tokenizer_opt == "gpt":
      tokenizer, embedding_model = get_gpt_tokenizer(**params)
    elif tokenizer_opt == "bert":
      tokenizer, embedding_model = get_BERT_tokenizer(**params)
      
    else:
      logger.error(f"{tokenizer_opt} is not among supperted tokenizer. Valid options are char, gpt and bert")
    
    return tokenizer, embedding_model


def get_gpt_tokenizer(src, cache_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(src,cache_dir=cache_dir)
    bos = '<|bos|>'
    eos = '<|eos|>'
    pad = '<|pad|>'
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}
    tokenizer.add_special_tokens(special_tokens_dict)
    embedding_model = GPT2LMHeadModel.from_pretrained(src,cache_dir=cache_dir)  # or any other checkpoint
    embedding_model.resize_token_embeddings(len(tokenizer))
    return tokenizer,embedding_model

def get_BERT_tokenizer(src,cache_dir):
  tokenizer = BertTokenizer.from_pretrained(src,cache_dir=cache_dir)
  bos = '<|bos|>'
  eos = '<|eos|>'
  special_tokens_dict = {'eos_token': eos, 'bos_token': bos}
  tokenizer.add_special_tokens(special_tokens_dict)
  embedding_model = BertModel.from_pretrained(src,cache_dir=cache_dir)
  embedding_model.resize_token_embeddings(len(tokenizer))
  return tokenizer,embedding_model

  
      
      
 