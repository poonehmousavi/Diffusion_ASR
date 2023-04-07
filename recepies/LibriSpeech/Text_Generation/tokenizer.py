from transformers import AutoTokenizer
import pandas as pd

def get_training_corpus(datasets):
    for start_idx in range(0, len(datasets), 1000):
        samples = datasets.iloc[start_idx : start_idx + 1000,-1]
        yield samples

old_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
datasets=pd.read_csv('./data_dir/train-clean-100.csv')

training_corpus = get_training_corpus(datasets)
tokenizer  = old_tokenizer.train_new_from_iterator(training_corpus, 1000)
tokenizer .save_pretrained("results/tokenizer/librispeecht-tokenizer")

example= datasets.iloc[1,-1]
print(tokenizer.vocab_size)
print(tokenizer.tokenize(example))
print(old_tokenizer.tokenize(example))



