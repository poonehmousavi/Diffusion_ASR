import torch
import torch.nn as nn


from lobes.RNN import RNNEncoder,RNNDecoder
from lobes.AutoEnoder import RNN_VAE
import yaml
import argparse
import os

from transformers import BertModel, BertTokenizer, AutoTokenizer
from utils.helper import dotdict

overfitting_number= 100
# importing module
import logging
import json
import csv

 

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)

# Creating an object
logger = logging.getLogger()

@torch.no_grad()
def sample(params_file, device):
    hparams={}
    logger.info("Start loading parameters file")
    with open(params_file, 'r') as file:
        config = yaml.safe_load(file)

    hparams= dotdict(config)

    # setting seed 
    seed= hparams.seed
    torch.manual_seed(seed)
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cexperiment_directory = os.path.join(hparams['output_folder'], str(seed))

    logger.info(f"Creating experiment directory: {cexperiment_directory}")
    create_experiment_directory(cexperiment_directory)


    #  Loading  Tokenizer and embedding model
    # cache_dir= os.path.join(cexperiment_directory, 'save')
    # tokenizer = BertTokenizer.from_pretrained(hparams['embedding_model_name'],cache_dir=cache_dir)
    # tokenizer.bos_token = tokenizer.cls_token
    # tokenizer.eos_token = tokenizer.sep_token
    # special_tokens= {'sos_idx':tokenizer.bos_token_id, 'eos_idx':tokenizer.eos_token_id, 'pad_idx':tokenizer.pad_token_id, 'unk_idx':tokenizer.unk_token_id}
    # embedding_model = BertModel.from_pretrained(hparams['embedding_model_name'],cache_dir=cache_dir)
    # hparams.encoder['latent_size'] = 2*hparams.encoder['latent_size']
    # encoder_model= RNNEncoder(device=device,embedding=embedding_model,**hparams.encoder)
    # decoder_model= RNNDecoder(device=device,embedding= embedding_model,**special_tokens,  **hparams.decoder)
    # model= RNN_VAE(encoder_model,decoder_model)

    tokenizer = AutoTokenizer.from_pretrained("results/tokenizer/librispeecht-tokenizer")

    bos = '<|bos|>'
    eos = '<|eos|>'
    pad = '<|pad|>'
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}
    tokenizer.add_special_tokens(special_tokens_dict) 
    special_tokens= {'sos_idx':tokenizer.bos_token_id, 'eos_idx':tokenizer.eos_token_id, 'pad_idx':tokenizer.pad_token_id, 'unk_idx':tokenizer.unk_token_id}
    # embedding_model = BertModel.from_pretrained(hparams['embedding_model_name'],cache_dir=cache_dir)
    # embedding_model = GPT2Model.from_pretrained("gpt2",cache_dir=cache_dir)
    # embedding_model = GPT2LMHeadModel.from_pretrained('gpt2',cache_dir=cache_dir)
    embedding_model = nn.Embedding(tokenizer.vocab_size+3, hparams.encoder['embedding_size'])
    hparams.encoder['latent_size'] = 2*hparams.encoder['latent_size']
    encoder_model= RNNEncoder(device=device,embedding=embedding_model,**hparams.encoder)
    decoder_model= RNNDecoder(device=device,embedding= embedding_model,**special_tokens,  **hparams.decoder)
    model= RNN_VAE(encoder_model,decoder_model)

    logger.info("Loading best model from save checkpoint")
    model = load_model(os.path.join(hparams['output_folder'],str(seed),'save','model.ckp'),model).to(device)
    samples_tokens=model.inference(19)
    samples= tokenizer.batch_decode(samples_tokens,skip_special_tokens=True)
    for sample in samples:
        print(sample)


def load_model(model_file_name,model):
    model.load_state_dict(torch.load(model_file_name))
    return model

def create_experiment_directory(experiment_directory):
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Text generation Experiment")
    parser.add_argument(
        "params",
        help='path to params file',
    )
    

    args = parser.parse_args()
    sample(
        args.params, 'cuda')