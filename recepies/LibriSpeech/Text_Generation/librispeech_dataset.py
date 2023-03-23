import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils.helper import dotdict
from torch.nn.utils.rnn import pad_sequence


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset."""

    def __init__(self, csv_file, root_dir, tokenizer):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio.
            tokenizer (tokenizer, ): tokenizer to be used
            on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.root_dir,
                                self.data.iloc[idx, 2])
        
        # TODO: load audio for phase 2 (conditioning on audio) 
        sig = path
        words = self.data.iloc[idx, -1]
        words = " ".join(words.split())

        tokens_list = self.tokenizer.encode(words)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        tokens_eos = torch.LongTensor(tokens_list[1:])
        tokens = torch.LongTensor(tokens_list[1:-1])


        return  sig, words, tokens , tokens_bos, tokens_eos
    
    def collate_fn(self, data):
        sig, words, tokens , tokens_bos, tokens_eos = zip(*data)
        # sig= [x['sig'] for x in batch]
        # words= [x['words'] for x in batch]
        # tokens= [x['tokens'] for x in batch]
        # tokens_bos= [x['tokens_bos'] for x in batch]
        # tokens_eos= [x['tokens_eos'] for x in batch]
        pad_value = self.tokenizer.pad_token_id
        
        sig_lens = torch.LongTensor([len(x) for x in sig])
        words_lens = torch.LongTensor([len(x) for x in words])
        tokens_lens = torch.LongTensor([len(x) for x in tokens])
        tokens_bos_lens = torch.LongTensor([len(x) for x in tokens_bos])
        tokens_eos_lens = torch.LongTensor([len(x) for x in tokens_eos])

        tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=pad_value)
        tokens_bos_padded = pad_sequence(tokens_bos, batch_first=True, padding_value=pad_value)
        tokens_eos_padded = pad_sequence(tokens_eos, batch_first=True, padding_value=pad_value)
        # padded_batch=[]

        padded_batch={'sig': (sig,sig_lens), 'words': (words,words_lens), 'tokens': (tokens_padded,tokens_lens) , 'tokens_bos': (tokens_bos_padded,tokens_bos_lens), 'tokens_eos':(tokens_eos_padded,tokens_eos_lens)}
        
        # for i in range(len(batch)):
        #    padded_batch.append({'sig': sig[i], 'words': words[i], 'tokens': tokens_padded[i] , 'tokens_bos': tokens_bos_padded[i], 'tokens_eos':tokens_eos_padded[i]})

        return padded_batch
        # return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]



        # return dotdict(sample)




#   xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
#   yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

#   return xx_pad, yy_pad, x_lens, y_lens

# class CustomPaddedBatch(PaddedBatch):
#     def __init__(self, examples, *args, **kwargs):
#         args.tokenizer
#         for k in ["tokens_bos", "tokens_eos", "tokens"]:
#             max_len = max([len(x[k]) for x in examples])
#             pad_value = args.tokenizer.pad_token_id
#             for example in examples:
#                 x = example[k]
#                 example[k] = torch.nn.functional.pad(
#                     x, [0, max_len - len(x)], value=pad_value
#                 )
#         super().__init__(examples, *args, **kwargs)   

# librispeech_dataset = LibriSpeechDataset(csv_file='librispeech_prepared/test-clean.csv',
#                                     root_dir='./data')

# for i in range(len(librispeech_dataset)):
#     sample = librispeech_dataset[i]