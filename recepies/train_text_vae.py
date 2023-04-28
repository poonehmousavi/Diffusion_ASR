import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import os
import argparse
from tqdm import tqdm
from jiwer import wer,cer
from functools import *
import sys


from data_utils.tokenizer import get_tokenizer
from data_utils.librispeech_dataset import LibriSpeechDataset
from utils.helper import *
from models.VAE import RNNVariationalAutoencoder, TransformerVariationalAutoencoder,AttentionTransformerVariationalAutoencoder
from models.transformer import generate_pad_mask,generate_square_subsequent_mask



overfitting_number= 10
# importing module
import logging
import json
import csv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%I:%M:%S",
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)

# Creating an object
logger = logging.getLogger()
 
best_epoch=0


def run(params_file, overfitting_test,load_from_pretrained,device, overrides):
    hparams={}
    logger.info("Start loading parameters file")
    with open(params_file, 'r') as file:
        config = yaml.safe_load(file)

    hparams= dotdict(config)
    hparams= resolve_overrides_params(hparams,overrides)
    # setting seed 
    seed= hparams.seed
    torch.manual_seed(seed)
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cexperiment_directory = os.path.join(hparams['output_folder'], str(seed))

    logger.info(f"Creating experiment directory: {cexperiment_directory}")
    create_experiment_directory(cexperiment_directory)


    #  Loading  Tokenizer and embedding model
    tokenizer, embedding = get_tokenizer(hparams['tokenizer_type'],**hparams['tokenizer'])
    word_embeddings= embedding.transformer.wte.weight
    special_tokens= {'bos_token_id':tokenizer.bos_token_id, 'eos_token_id':tokenizer.eos_token_id, 'pad_token_id':tokenizer.pad_token_id, 'unk_token_id':tokenizer.unk_token_id}
    
    # Loading Dataset and Dataloader
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
    

    # Loading Model
    if hparams['model_type'].lower() == 'rnn':
        model = RNNVariationalAutoencoder(**special_tokens, **hparams['model'], embedding_weights=word_embeddings,vocab_size =len(tokenizer), device=device)
    elif  hparams['model_type'].lower() == 'transformer':
        model= TransformerVariationalAutoencoder(**special_tokens, **hparams['model'], embedding_weights=word_embeddings,vocab_size =len(tokenizer), device=device)
    elif hparams['model_type'].lower() == 'transformer_with_attention':
        model= AttentionTransformerVariationalAutoencoder(**special_tokens, **hparams['model'], embedding_weights=word_embeddings,vocab_size =len(tokenizer), device=device)

    else:
        logger.error(f"{hparams['model_type']} is not among supperted model types. Supported models are rnn, transformer and transformer_with_attention")


    if load_from_pretrained:
        logger.info("Loading from save checkpoint")
        model = load_model(os.path.join(hparams['output_folder'],str(seed),'save','model.ckp'),model)
    # Define Optimizer
    if hparams['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(hparams['lr']))
    elif hparams['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams['lr']))
    elif hparams['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(hparams['lr']))
    else:
        logger.error(f"{hparams['optimizer']} is not among supperted optimizer. Supported models are SGD and Adam")


    logger.debug("Start training ....")
    loss_module = partial(vae_loss, ignore_index = tokenizer.pad_token_id, variational_beta = hparams['variational_beta'])

    train(model, optimizer, train_loader, valid_loader, hparams,tokenizer,loss_module, device)


    logger.info("Loading best model from save checkpoint")
    model = load_model(os.path.join(hparams['output_folder'],str(seed),'save','model.ckp'),model)
    eval(model ,test_loader, loss_module ,hparams,tokenizer, device)

# def get_emebbding():
def vae_loss(recon_x, x, mu, logvar,ignore_index, variational_beta):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    batch_size, seq_len,logits_dim= recon_x.shape

    recon_loss = F.cross_entropy(recon_x.view(-1,logits_dim), x.view(-1), reduction='sum', ignore_index=ignore_index)
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + variational_beta * kldivergence , recon_loss, kldivergence

def train(model, optimizer, train_loader, valid_loader, hparams,tokenizer,loss_module, device='cuda'):
    model =model.to(device)
    best_loss = float('inf')
   
    create_experiment_directory(os.path.join(hparams['output_folder'],str(hparams.seed),'save'))
    model =model.to(device)
    step = 0
    
    # Training loop
    for epoch in tqdm(range(hparams['number_of_epochs'])):
        # Set model to train mode
        tr_losses=[]
        NLL_losses=[]
        KL_losses=[]
        for iteration,batch in enumerate(tqdm(train_loader)):
            model.train()
            loss, NLL_loss, KL_loss =train_batch(model, batch,optimizer,loss_module,tokenizer,hparams,device)
            tr_losses.append(loss)
            NLL_losses.append(NLL_loss)
            KL_losses.append(KL_loss)

            if iteration % hparams['print_every'] == 0 or iteration+1 == len(train_loader):
                    logger.info("Training Batch %04d/%i, ELBO Loss %9.4f , NLL-Loss %9.4f, KL-Loss %9.4f"
                          % (iteration, len(train_loader)-1, loss,NLL_loss, KL_loss))
            step += 1
        logger.info("Training Epoch %02d/%i, ELBO Loss %9.4f , NLL-Loss %9.4f, KL-Loss %9.4f" % (epoch, hparams['number_of_epochs'], np.mean(tr_losses),np.mean(NLL_losses),np.mean(KL_losses)))
                  
        valid_losses=[]
        valid_NLL_losses=[]
        valid_KL_losses=[]
        references=[]
        hypothesises=[]
        for iteration,batch in enumerate(tqdm(valid_loader)):
            model.eval()
            loss, NLL_loss, KL_loss, hypothesis,reference= eval_batch(model, batch,loss_module,hparams,tokenizer, device)
            valid_losses.append(loss)
            valid_NLL_losses.append(NLL_loss)
            valid_KL_losses.append(KL_loss)

            hypothesises.extend(hypothesis)
            references.extend(reference)
           

        wer_score = wer(references,hypothesises)*100
        cer_score= cer(references,hypothesises)*100
        logger.info("Valid Epoch %02d/%i, ELBO Loss %9.4f , NLL-Loss %9.4f, KL-Loss %9.4f, Valid WER  %9.4f, Valid CER %9.4f" % (epoch, hparams['number_of_epochs'], np.mean(valid_losses),np.mean(valid_NLL_losses),np.mean(valid_KL_losses),wer_score,cer_score))
        
        # save loss stats
        log_file = open(os.path.join(hparams['output_folder'],str(hparams.seed),hparams['train_logs']), "a")
        # json.dump({'Epoch':epoch, 'train loss': np.mean(tr_losses), 'valid loss':np.mean(valid_losses), 'valid WER': wer_score , 'valid_cer': cer_score}, log_file, indent = 6)
        log_file.write(f"Epoch: {epoch}, train loss: {np.mean(tr_losses)}, train_NLL: {np.mean(NLL_losses)}, train_KL: {np.mean(KL_losses)}, valid loss: {np.mean(valid_losses)}, valid NLL: {np.mean(valid_NLL_losses)}, Valid KL: {np.mean(valid_KL_losses)},  valid WER: {wer_score} , valid_cer: {cer_score}\n")
        log_file.close()
        
        # save best model based on WER
        if (np.mean(tr_losses) < best_loss):
            logger.info("saving best model into save checkpoint")
            global best_epoch
            best_epoch = epoch
            best_loss = np.mean(tr_losses)
            save_model(os.path.join(hparams['output_folder'],str(hparams.seed),'save','model.ckp'),model)
            save_model(os.path.join(hparams['output_folder'],str(hparams.seed),'save','encoder.ckp'),model.encoder)
            save_model(os.path.join(hparams['output_folder'],str(hparams.seed),'save','decoder.ckp'),model.decoder)
    


def train_batch(model, batch,optimizer,loss_module,tokenizer,hparams,device): 
    batch= dotdict(batch)
          
    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
    input_ids, input_ids_lens = batch['input_ids']
    dec_input_ids, dec_input_ids_lens = batch['dec_input_ids']
    labels, labels_lens = batch['labels']
    input_ids,dec_input_ids,labels= input_ids.to(device),dec_input_ids.to(device),labels.to(device)
    batch_size, seq_len= input_ids.shape
            
    ## Step 2: Run the model on the input data
    # vae reconstruction
    if hparams['model_type'].lower() == 'rnn':
        text_batch_recon, latent_mu, latent_logvar = model(input_ids, input_ids_lens,dec_input_ids, dec_input_ids_lens)
    elif hparams['model_type'].lower() == 'transformer':
        trg_lookahead_mask= generate_square_subsequent_mask(dec_input_ids.shape[1]).to(device)
        trg_pad_mask= generate_pad_mask(dec_input_ids, tokenizer.pad_token_id)
        src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
        text_batch_recon, latent_mu, latent_logvar = model(input_ids,dec_input_ids,src_pad_mask,trg_pad_mask,trg_lookahead_mask )
    elif hparams['model_type'].lower() == 'transformer_with_attention':
        trg_lookahead_mask= generate_square_subsequent_mask(dec_input_ids.shape[1]).to(device)
        trg_pad_mask= generate_pad_mask(dec_input_ids, tokenizer.pad_token_id)
        src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
        text_batch_recon, latent_mu, latent_logvar = model(input_ids,dec_input_ids,src_pad_mask,trg_pad_mask,trg_lookahead_mask )


            
    ## Step 3: Calculate the loss reconstruction error
    loss, NLL_loss ,KL_loss= loss_module(text_batch_recon, labels, latent_mu, latent_logvar)

            
    ## Step 4: Perform backpropagation
    # Before calculating the gradients, we need to ensure that they are all zero. 
    # The gradients would not be overwritten, but actually added to the existing ones.
    optimizer.zero_grad() 
    # Perform backpropagation
    loss.backward()
            
    ## Step 5: Update the parameters
    optimizer.step()

    return loss.item()/batch_size, NLL_loss.item()/ batch_size, KL_loss.item()/batch_size

@torch.no_grad()
def eval(model ,data_loader, loss_module ,hparams,tokenizer, device='cuda'):
   
    model =model.to(device)
            
    losses=[]
    NLL_losses=[]
    KL_losses=[]
    references=[]
    hypothesises=[]
    for iteration,batch in enumerate(tqdm(data_loader)):
        loss, NLL_loss, KL_loss, hypothesis,reference= eval_batch(model, batch,loss_module,hparams,tokenizer, device)
        losses.append(loss)
        NLL_losses.append(NLL_loss)
        KL_losses.append(KL_loss)

        hypothesises.extend(hypothesis)
        references.extend(reference)
           

    wer_score = wer(references,hypothesises)*100
    cer_score= cer(references,hypothesises)*100
    
    logger.info("Test, ELBO Loss %9.4f , NLL-Loss %9.4f, KL-Loss %9.4f, Test WER  %9.4f, Test CER %9.4f" % (np.mean(losses),np.mean(NLL_losses),np.mean(KL_losses),wer_score,cer_score))
        
    # save loss stats
    log_file = open(os.path.join(hparams['output_folder'],str(hparams.seed),hparams['train_logs']), "a")
    # json.dump({'Epoch':epoch, 'train loss': np.mean(tr_losses), 'valid loss':np.mean(valid_losses), 'valid WER': wer_score , 'valid_cer': cer_score}, log_file, indent = 6)
    log_file.write(f"Test loss: {np.mean(losses)}, test NLL: {np.mean(NLL_losses)}, test KL: {np.mean(losses)},  test WER: {wer_score} , test CER: {cer_score}\n")
    log_file.close()
        
    
    wers=[wer(references[i],hypothesises[i])*100 for i in range(len(references))]
    cers=[cer(references[i],hypothesises[i])*100 for i in range(len(references))]

    header = ['reference', 'predicted', 'wer','cer']
    # save wer file
    with open(os.path.join(hparams['output_folder'],str(hparams.seed),hparams['test_wer']), 'w') as f:
        wer_data= [{'refernce': reference, 'hypothesis': hypothesises, 'wer': wer, 'cer': cer} for reference, hypothesises, wer, cer in zip(references, hypothesises,wers,cers)]
        for dic in wer_data:
            json.dump(dic, f) 
            f.write("\n")

        # writer = csv.writer(f,delimiter=',')
        # writer.writerow(i for i in header)
        # writer.writerows(zip(references, hypothesises,wers, cers ))



@torch.no_grad()
def eval_batch(model, batch,loss_module,hparams, tokenizer, device):
    batch= dotdict(batch)
          
    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
    input_ids, input_ids_lens = batch['input_ids']
    dec_input_ids, dec_input_ids_lens = batch['dec_input_ids']
    labels, labels_lens = batch['labels']
    input_ids,dec_input_ids,labels= input_ids.to(device),dec_input_ids.to(device),labels.to(device)
    batch_size, seq_len= input_ids.shape
            
    ## Step 2: Run the model on the input data
    # vae reconstruction
    if hparams['model_type'].lower() == 'rnn':
        text_batch_recon, latent_mu, latent_logvar = model(input_ids, input_ids_lens,dec_input_ids, dec_input_ids_lens)
        hyp = model.generate(input_ids, input_ids_lens,mode = hparams['decoder_search'])
    elif hparams['model_type'].lower() == 'transformer':
        trg_lookahead_mask= generate_square_subsequent_mask(dec_input_ids.shape[1]).to(device)
        trg_pad_mask= generate_pad_mask(dec_input_ids, tokenizer.pad_token_id)
        src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
        text_batch_recon, latent_mu, latent_logvar = model(input_ids,dec_input_ids,src_pad_mask,trg_pad_mask,trg_lookahead_mask )
        hyp = model.generate(input_ids,src_pad_mask, mode = hparams['decoder_search'])
    elif hparams['model_type'].lower() == 'transformer_with_attention':
        trg_lookahead_mask= generate_square_subsequent_mask(dec_input_ids.shape[1]).to(device)
        trg_pad_mask= generate_pad_mask(dec_input_ids, tokenizer.pad_token_id)
        src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
        text_batch_recon, latent_mu, latent_logvar = model(input_ids,dec_input_ids,src_pad_mask,trg_pad_mask,trg_lookahead_mask )
        hyp = model.generate(input_ids,src_pad_mask, mode = hparams['decoder_search'])


            
    ## Step 3: Calculate the loss reconstruction error
    loss, NLL_loss ,KL_loss= loss_module(text_batch_recon, labels, latent_mu, latent_logvar)

    
    hypothesis= tokenizer.batch_decode(hyp,skip_special_tokens=True)
    reference= tokenizer.batch_decode(input_ids,skip_special_tokens=True)
    
    return loss.item()/batch_size, NLL_loss.item()/ batch_size, KL_loss.item()/batch_size ,hypothesis,reference


    





if __name__ == "__main__":
    arg_list = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run Text generation Experiment")
    parser.add_argument(
        "params",
        help='path to params file',
    )
    parser.add_argument(
        "--device",
        help='device',
        default='cuda'
    )
    
    parser.add_argument("--overfitting",action='store_true', help="in overfitting test mode",default=False)
    parser.add_argument("--load_from_pretrained",action='store_true', help="if load frm pretrained and continue training from that point",default=False)


    # args = parser.parse_args()
    run_opts, overrides = parser.parse_known_args(arg_list)
    run(
        run_opts.params, run_opts.overfitting, run_opts.load_from_pretrained, run_opts.device, overrides)
