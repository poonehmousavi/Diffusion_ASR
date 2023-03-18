import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import timeit
import random
import datetime
from torch.utils.data import Dataset, DataLoader
from librispeech_dataset import LibriSpeechDataset
from lobes.RNN_AE import RNNEncoder,RNNDecoder,RNN_AE
import yaml
import argparse
import os
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from utils.helper import dotdict
from transformers import DataCollatorWithPadding




# importing module
import logging
 

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
 
 

def run(params_file, device):
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
    cache_dir= os.path.join(cexperiment_directory, 'save')
    tokenizer = BertTokenizer.from_pretrained(hparams['embedding_model_name'],cache_dir=cache_dir)
    embedding_model = BertModel.from_pretrained(hparams['embedding_model_name'],cache_dir=cache_dir)
    encoder_model= RNNEncoder(device, dotdict(hparams.encoder),embedding_model)
    decoder_model= RNNDecoder(device, dotdict(hparams.decoder),embedding_model)
    model= RNN_AE(encoder_model,decoder_model)


    train_set = LibriSpeechDataset(csv_file=os.path.join(cexperiment_directory, hparams['train_csv']), root_dir=hparams['data_folder'],tokenizer=tokenizer)
    hparams["train_loader_kwargs"]["collate_fn"] = train_set.collate_fn
    train_loader = torch.utils.data.DataLoader(train_set, **hparams['train_loader_kwargs'])
    
    valid_set = LibriSpeechDataset(csv_file=os.path.join(cexperiment_directory, hparams['valid_csv']), root_dir=hparams['data_folder'],tokenizer=tokenizer)
    hparams["valid_loader_kwargs"]["collate_fn"] = valid_set.collate_fn
    valid_loader = torch.utils.data.DataLoader(valid_set, **hparams['valid_loader_kwargs'])

    test_set = LibriSpeechDataset(csv_file=os.path.join(cexperiment_directory, hparams['test_csv']), root_dir=hparams['data_folder'],tokenizer=tokenizer)
    hparams["test_loader_kwargs"]["collate_fn"] = test_set.collate_fn
    test_loader = torch.utils.data.DataLoader(test_set, **hparams['test_loader_kwargs'])

    
    loss_module = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams['lr'])

    fit(model,optimizer,train_loader,valid_loader,loss_module, hparams['number_of_epochs'],device)


# def get_emebbding():

def fit(model, optimizer, train_loader, valid_loader, loss_module, num_epochs,device='cuda'):
    # Set model to train mode
   
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train() 
        for i,batch in enumerate(train_loader):
            train_batch(model, batch, device,loss_module,optimizer)
            
        model.eval()
        for i,batch in enumerate(valid_loader):
            eval_batch(model, batch, device)

def train_batch(model, batch, device,loss_module,optimizer):
    batch= dotdict(batch)
          
    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
    tokens, tokens_lens = batch.tokens
    tokens_bos, tokens_bos_lens = batch.tokens_bos
    tokens_eos, tokens_eos_lens = batch.tokens_eos
    tokens,tokens_bos,tokens_eos= tokens.to(device),tokens_bos.to(device),tokens_eos.to(device)
            
    ## Step 2: Run the model on the input data
    preds = model(tokens,tokens_bos)
    preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
    ## Step 3: Calculate the loss
    loss = loss_module(preds, tokens_eos)
            
    ## Step 4: Perform backpropagation
    # Before calculating the gradients, we need to ensure that they are all zero. 
    # The gradients would not be overwritten, but actually added to the existing ones.
    optimizer.zero_grad() 
    # Perform backpropagation
    loss.backward()
            
    ## Step 5: Update the parameters
    optimizer.step()

    return model
def eval_batch(model, batch, device,loss_module):
    batch= dotdict(batch)
          
    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
    tokens, tokens_lens = batch.tokens
    tokens_bos, tokens_bos_lens = batch.tokens_bos
    tokens_eos, tokens_eos_lens = batch.tokens_eos
    tokens,tokens_bos,tokens_eos= tokens.to(device),tokens_bos.to(device),tokens_eos.to(device)
            
    ## Step 2: Run the model on the input data
    preds = model(tokens,tokens_bos)
    preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
    ## Step 3: Calculate the loss
    loss = loss_module(preds, tokens_eos)


    

def create_experiment_directory(experiment_directory):
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)



def save_models(self, encoder_file_name, decoder_file_name):
    torch.save(self.encoder.state_dict(), encoder_file_name)
    torch.save(self.decoder.state_dict(), decoder_file_name)

def load_models(self, encoder_file_name, decoder_file_name):
    self.encoder.load_state_dict(torch.load(encoder_file_name))
    self.decoder.load_state_dict(torch.load(decoder_file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Text generation Experiment")
    parser.add_argument(
        "params",
        help='path to params file',
    )

    args = parser.parse_args()
    run(
        args.params, 'cuda')