import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from librispeech_dataset import LibriSpeechDataset
from lobes.RNN import RNNEncoder,RNNDecoder
from lobes.AutoEnoder import RNN_AE
import yaml
import argparse
import os
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from utils.helper import dotdict
from transformers import DataCollatorWithPadding
from torch.utils.data.sampler import SubsetRandomSampler
from jiwer import wer,cer

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
 
best_epoch=0


def run(params_file, device,overfitting_test=False):
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
    special_tokens= {'sos_idx':tokenizer.cls_token_id, 'eos_idx':tokenizer.sep_token_id, 'pad_idx':tokenizer.pad_token_id, 'unk_idx':tokenizer.unk_token_id}
    embedding_model = BertModel.from_pretrained(hparams['embedding_model_name'],cache_dir=cache_dir)
    
    encoder_model= RNNEncoder(device=device,embedding=embedding_model,**hparams.encoder)
    decoder_model= RNNDecoder(device=device,embedding= embedding_model,**special_tokens,  **hparams.decoder)
    model= RNN_AE(encoder_model,decoder_model)


    train_set = LibriSpeechDataset(csv_file=os.path.join(hparams['data_folder'], hparams['train_csv']), root_dir=hparams['data_folder'],tokenizer=tokenizer)
    hparams["train_loader_kwargs"]["collate_fn"] = train_set.collate_fn
    if overfitting_test:
        train_set = torch.utils.data.random_split(train_set, [overfitting_number, len(train_set)-overfitting_number])[0]
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, **hparams['train_loader_kwargs'])


    
    valid_set = LibriSpeechDataset(csv_file=os.path.join(hparams['data_folder'], hparams['valid_csv']), root_dir=hparams['data_folder'],tokenizer=tokenizer)
    hparams["valid_loader_kwargs"]["collate_fn"] = valid_set.collate_fn
    if overfitting_test:
        valid_set = torch.utils.data.random_split(valid_set, [overfitting_number, len(valid_set)-overfitting_number])[0]
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True,**hparams['valid_loader_kwargs'])

    test_set = LibriSpeechDataset(csv_file=os.path.join(hparams['data_folder'], hparams['test_csv']), root_dir=hparams['data_folder'],tokenizer=tokenizer)
    hparams["test_loader_kwargs"]["collate_fn"] = test_set.collate_fn
    if overfitting_test:
        test_set = torch.utils.data.random_split(test_set, [overfitting_number, len(test_set)-overfitting_number])[0]
    test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True,**hparams['test_loader_kwargs'])

    
    loss_module = torch.nn.NLLLoss(ignore_index=special_tokens['pad_idx'], reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=hparams['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    fit(model,optimizer,train_loader,valid_loader,loss_module, hparams,tokenizer,device=device)
    model= RNN_AE(encoder_model,decoder_model)
    logger.info("Loading best model from save checkpoint")
    model = load_model(os.path.join(hparams['output_folder'],str(seed),'save','model.ckp'),model)
    eval(model ,test_loader, loss_module ,hparams,tokenizer, device)
    # eval(model ,valid_loader, loss_module ,hparams, tokenizer, device)

# def get_emebbding():

def fit(model, optimizer, train_loader, valid_loader, loss_module, hparams,tokenizer, device='cuda'):
    best_wer = float('inf')
    create_experiment_directory(os.path.join(hparams['output_folder'],str(hparams.seed),'save'))
    model =model.to(device)
    # Training loop
    for epoch in tqdm(range(hparams['number_of_epochs'])):
        # Set model to train mode
       
        tr_losses=[]
        for iteration,batch in enumerate(tqdm(train_loader)):
            model, loss=train_batch(model, batch, device,loss_module,optimizer)
            tr_losses.append(loss.item())
            if iteration % hparams['print_every'] == 0 or iteration+1 == len(train_loader):
                    logger.info("Training Batch %04d/%i, Loss %9.4f"
                          % (iteration, len(train_loader)-1, loss.item()))
        logger.info("Training Epoch %02d/%i, ,Mean NLL Loss %9.4f" % (epoch, hparams['number_of_epochs'], np.mean(tr_losses)))
        
            
        valid_losses=[]
        references=[]
        hypothesises=[]
        for iteration,batch in enumerate(tqdm(valid_loader)):
            model, loss, hypothesis,reference= eval_batch(model, batch, device, loss_module,tokenizer)
            valid_losses.append(loss.item())
            hypothesises.extend(hypothesis)
            references.extend(reference)

        wer_score = wer(references,hypothesises)*100
        cer_score= cer(references,hypothesises)*100
        logger.info("Valid Epoch %02d/%i,Mean NLL Loss %9.4f, Valid WER  %9.4f, Valid CER %9.4f" % (epoch, hparams['number_of_epochs'], np.mean(valid_losses),wer_score,cer_score))
        
        # save loss stats
        log_file = open(os.path.join(hparams['output_folder'],hparams['train_logs']), "a")
        # json.dump({'Epoch':epoch, 'train loss': np.mean(tr_losses), 'valid loss':np.mean(valid_losses), 'valid WER': wer_score , 'valid_cer': cer_score}, log_file, indent = 6)
        log_file.write(f"Epoch: {epoch}, train loss: {np.mean(tr_losses)}, valid loss: {np.mean(valid_losses)}, valid WER: {wer_score} , valid_cer: {cer_score}\n")
        log_file.close()
        
        # save best model based on WER
        if (wer_score < best_wer):
            logger.info("saving best model into save checkpoint")
            global best_epoch
            best_epoch = epoch
            best_wer = wer_score
            save_model(os.path.join(hparams['output_folder'],str(hparams.seed),'save','model.ckp'),model)
            save_model(os.path.join(hparams['output_folder'],str(hparams.seed),'save','encoder.ckp'),model.encoder)
            save_model(os.path.join(hparams['output_folder'],str(hparams.seed),'save','decoder.ckp'),model.decoder)



def eval(model ,test_loader, loss_module ,hparams,tokenizer, device='cuda'):
   
    model =model.to(device)
            
    test_losses=[]
    references=[]
    hypothesises=[]
    for iteration,batch in enumerate(tqdm(test_loader)):
        model, loss, hypothesis,reference= eval_batch(model, batch, device, loss_module,tokenizer)
        test_losses.append(loss.item())
        hypothesises.extend(hypothesis)
        references.extend(reference)

    logger.info("Test ,Mean NLL Loss %9.4f, WER  %9.4f" % ( np.mean(test_losses)*100,wer(references,hypothesises)*100))
    # save loss stats
    log_file = open(os.path.join(hparams['output_folder'],hparams['train_logs']), "a")
    # json.dump({'Epoch loaded':best_epoch, 'test loss': np.mean(test_losses), 'test WER': wer(references,hypothesises)*100 , 'test_cer': cer(references,hypothesises)*100}, log_file, indent = 6)
    log_file.write(f'Epoch loaded: {best_epoch}, test loss: {np.mean(test_losses)}, test WER: {wer(references,hypothesises)*100} , test_cer: {cer(references,hypothesises)*100}')

    log_file.close()
    
    wers=[wer(references[i],hypothesises[i])for i in range(len(references))]
    header = ['reference', 'predicted', 'wer']
    # save wer file
    with open(os.path.join(hparams['output_folder'],str(hparams.seed),hparams['test_wer']), 'w') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(i for i in header)
        writer.writerows(zip(references, hypothesises,wers ))



def train_batch(model, batch, device,loss_module,optimizer):
    model.train() 
    batch= dotdict(batch)
    
          
    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
    tokens, tokens_lens = batch.tokens
    tokens_bos, tokens_bos_lens = batch.tokens_bos
    tokens_eos, tokens_eos_lens = batch.tokens_eos
    tokens,tokens_bos,tokens_eos= tokens.to(device),tokens_bos.to(device),tokens_eos.to(device)
            
    ## Step 2: Run the model on the input data
    preds = model(tokens,tokens_bos,tokens_lens, tokens_bos_lens)
    preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
    ## Step 3: Calculate the loss
    logp= nn.functional.log_softmax(preds, dim=-1)
    logp = logp.view(-1, logp.size(2))
            
    target = tokens_eos[:, :torch.max(tokens_eos_lens).item()].contiguous().view(-1)
    loss = loss_module(logp, target)
    
    ## Step 4: Perform backpropagation
    # Before calculating the gradients, we need to ensure that they are all zero. 
    # The gradients would not be overwritten, but actually added to the existing ones.
    optimizer.zero_grad() 
    # Perform backpropagation
    loss.backward()
            
    ## Step 5: Update the parameters
    optimizer.step()

    return model,loss

def eval_batch(model, batch, device,loss_module,tokenizer):
    model.eval()
    batch= dotdict(batch)
    
          
    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
    tokens, tokens_lens = batch.tokens
    tokens_bos, tokens_bos_lens = batch.tokens_bos
    tokens_eos, tokens_eos_lens = batch.tokens_eos
    tokens,tokens_bos,tokens_eos= tokens.to(device),tokens_bos.to(device),tokens_eos.to(device)
            
    ## Step 2: Run the model on the input data
    preds = model(tokens,tokens_bos,tokens_lens, tokens_bos_lens)
    preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
    ## Step 3: Calculate the loss
    logp= nn.functional.log_softmax(preds, dim=-1)
    logp = logp.view(-1, logp.size(2))
            
    target = tokens_eos[:, :torch.max(tokens_eos_lens).item()].contiguous().view(-1)
    loss = loss_module(logp, target)

    hyp = model.generate(tokens,tokens_lens)
   
    # target_tokens=[]
    # target_tokens=[]
    # for i in range(tokens.shape[0]):
    #     target_tokens.append(tokens[i,:tokens_lens[i]])
    hypothesis= tokenizer.batch_decode(hyp,skip_special_tokens=True)
    reference= tokenizer.batch_decode(tokens,skip_special_tokens=True)

    
    


    
    return model,loss,hypothesis,reference


    

def create_experiment_directory(experiment_directory):
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)


def save_model(model_file_name,model):
    torch.save(model.state_dict(), model_file_name)


def load_model(model_file_name,model):
    model.load_state_dict(torch.load(model_file_name))
    return model


# def save_models(self, encoder_file_name, decoder_file_name, model):
#     torch.save(self.encoder.state_dict(), encoder_file_name)
#     torch.save(self.decoder.state_dict(), decoder_file_name)


# def load_models(self, encoder_file_name, decoder_file_name):
#     self.encoder.load_state_dict(torch.load(encoder_file_name))
#     self.decoder.load_state_dict(torch.load(decoder_file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Text generation Experiment")
    parser.add_argument(
        "params",
        help='path to params file',
    )
    parser.add_argument("--overfitting",action='store_true', help="in overfitting test mode")

    args = parser.parse_args()
    run(
        args.params, 'cuda', args.overfitting)