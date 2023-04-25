import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset."""

    def __init__(self, csv_file, root_dir, tokenizer,max_sequence_length=50):
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
        self.max_sequence_length = max_sequence_length
        

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
        l= words.split()
        if len(l) > self.max_sequence_length:
            l=l[:self.max_sequence_length]

        words = " ".join(l)

        tokens_list = self.tokenizer.encode(words,add_special_tokens=False)
        tokens_bos = torch.LongTensor([self.tokenizer.bos_token_id]+tokens_list)
        tokens_eos = torch.LongTensor(tokens_list+[self.tokenizer.eos_token_id])
        tokens = torch.LongTensor(tokens_list)

        return {
            'input_ids': tokens,
            'labels': tokens_eos,
            'dec_input_ids': tokens_bos,
            'audio_filepath': path
        }



    def collate_fn(self,  features: dict):
      input_ids, labels, dec_input_ids, audio_filepaths = [], [], [], []
      for f in features:
            audio_filepaths.append(f['audio_filepath'])
            input_ids.append(f['input_ids'])
            labels.append(f['labels'])
            dec_input_ids.append(f['dec_input_ids'])

      pad_value = self.tokenizer.pad_token_id
      
      audio_filepaths_lens = torch.LongTensor([len(x) for x in audio_filepaths])
      input_ids_lens = torch.LongTensor([len(x) for x in input_ids])
      labels_lens = torch.LongTensor([len(x) for x in labels])
      dec_input_ids_lens = torch.LongTensor([len(x) for x in dec_input_ids])

      input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_value)
      labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_value)
      dec_input_ids_padded = pad_sequence(dec_input_ids, batch_first=True, padding_value=pad_value)

      padded_batch={'audio_filepaths': (audio_filepaths,audio_filepaths_lens), 'dec_input_ids': (dec_input_ids_padded,dec_input_ids_lens), 'input_ids': (input_ids_padded,input_ids_lens) , 'labels': (labels_padded,labels_lens)}

      return padded_batch

if __name__ == "__main__":
   params={'src': 'gpt2', 'cache_dir': './cache'}
   
   from tokenizer import get_tokenizer
   tokenizer, embedding = get_tokenizer('gpt',**params)
   
   train_set = LibriSpeechDataset(csv_file="./data_dir/train-clean-100.csv", root_dir='./',tokenizer=tokenizer)
   train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, collate_fn= train_set.collate_fn ,batch_size=32, shuffle=True )
   batch = next(iter(train_loader))
   print("Exit successfully!")



