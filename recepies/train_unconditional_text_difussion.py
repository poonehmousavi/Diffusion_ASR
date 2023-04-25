import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging
import sys


from data_utils.tokenizer import get_tokenizer
from data_utils.librispeech_dataset import LibriSpeechDataset
from utils.helper import *
from models.VAE import RNNVariationalAutoencoder, TransformerVariationalAutoencoder,AttentionTransformerVariationalAutoencoder
from models.transformer import generate_pad_mask,generate_square_subsequent_mask,Transformer_Diffusion
from difussion.ddpm import Diffusion
from models.UNET import UNet_1D
from jiwer import wer,cer


overfitting_number= 3
best_epoch=0

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
 

def run(params_file, overfitting_test,device,overrides):
    best_loss = float('inf')
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
    if hparams['ae_model_type'].lower() == 'rnn':
        ae_model = RNNVariationalAutoencoder(**special_tokens, **hparams['ae_model'], embedding_weights=None,vocab_size =len(tokenizer), device=device)
    elif  hparams['ae_model_type'].lower() == 'transformer':
        ae_model= TransformerVariationalAutoencoder(**special_tokens, **hparams['ae_model'], embedding_weights=None,vocab_size =len(tokenizer), device=device)
    elif hparams['ae_model_type'].lower() == 'transformer_with_attention':
        ae_model= AttentionTransformerVariationalAutoencoder(**special_tokens, **hparams['ae_model'], embedding_weights=None,vocab_size =len(tokenizer), device=device)
    else:
        logger.error(f"{hparams['ae_model_type']} is not among supperted model types. Supported models are rnn, transformer and transformer_with_attention")
    
    
    hparams['ae_model_checkoint']=  os.path.join(data_root,hparams['ae_model_checkoint'])
    ae_model = load_model(hparams['ae_model_checkoint'],ae_model)



    if hparams['diff_model_type'].lower() == 'transformer':
        diffusion_model = Transformer_Diffusion(**hparams['diffusion_model'])
    elif hparams['diff_model_type'].lower() == 'unet_1d':
        diffusion_model =  UNet_1D(**hparams['diffusion_model'],device=device)
    else:
        logger.error(f"{hparams['diff_model_type']} is not among supperted model types for diffusion. Supported models are  transformer and unet_1d")


    # Define Optimizer
    if hparams['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(diffusion_model.parameters(), lr=float(hparams['lr']))
    elif hparams['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=float(hparams['lr']))
    elif hparams['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=float(hparams['lr']))
    else:
        logger.error(f"{hparams['optimizer']} is not among supperted optimizer. Supported models are SGD , Adam and AdamW")    
    
    
    create_experiment_directory(os.path.join(hparams['output_folder'],str(hparams.seed),'save'))
    create_experiment_directory(os.path.join(hparams['output_folder'],str(hparams.seed),'samples'))

    mse = nn.MSELoss()
    diffusion = Diffusion(**hparams['diffusion'], device=device)
    diffusion_model =diffusion_model.to(device)
    ae_model = ae_model.to(device)
    ae_model.eval()
    

    for epoch in range(hparams['number_of_epochs']):
        logging.info(f"Starting epoch {epoch}:")
        train_loss=[]
        for i, batch in enumerate(tqdm(train_loader)):
            diffusion_model.train()
            input_ids, input_ids_lens = batch['input_ids']
            batch_size, seq_len= input_ids.shape
            latent=generate_latent(ae_model,input_ids, input_ids_lens, tokenizer, hparams,device)
            # sample timestep and generate noisy latents
            t = diffusion.sample_timesteps(batch_size).to(device)
            x_t, noise = diffusion.noise_embedding(latent, t)
            # pass data to diffusion model to predict the noise
            predicted_noise = diffusion_model(x_t, t)

            # Calculate the loss
            loss = mse(noise, predicted_noise)

            # calculate gradient and update parameter 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        logger.info("Training Epoch %02d/%i, Loss %9.4f" % (epoch, hparams['number_of_epochs'], np.mean(train_loss)))

        valid_loss=[]
        references=[]
        hypothesises=[]
        diffusion_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_loader)):
                input_ids, input_ids_lens = batch['input_ids']
                labels, labels_lens = batch['labels']
                batch_size, seq_len= input_ids.shape
                input_ids,labels= input_ids.to(device),labels.to(device)
                latent=generate_latent(ae_model,input_ids, input_ids_lens, tokenizer, hparams,device)
                # sample timestep and generate noisy latents
                t = diffusion.sample_timesteps(batch_size).to(device)
                x_t, noise = diffusion.noise_embedding(latent, t)
                # pass data to diffusion model to predict the noise
                predicted_noise = diffusion_model(x_t, t)

                # Calculate the loss
                loss = mse(noise, predicted_noise)
                denoised_latent = diffusion.sample(diffusion_model, x=latent)
                valid_loss.append(loss.item())

                hyp= generate_txt(ae_model,denoised_latent.squeeze(1),tokenizer,hparams,device,input_ids=input_ids)
                hypothesis= tokenizer.batch_decode(hyp,skip_special_tokens=True)
                reference= tokenizer.batch_decode(input_ids,skip_special_tokens=True)
                hypothesises.extend(hypothesis)
                references.extend(reference)


            wer_score = wer(references,hypothesises)*100
            cer_score= cer(references,hypothesises)*100
            logger.info("Valid Epoch %02d/%i,Loss %9.4f , Valid WER  %9.4f, Valid CER %9.4f" % (epoch, hparams['number_of_epochs'], np.mean(valid_loss),wer_score,cer_score))
            # save loss stats
            log_file = open(os.path.join(hparams['output_folder'],str(hparams.seed),hparams['train_logs']), "a")
            log_file.write(f"Epoch: {epoch}, train loss: {np.mean(train_loss)}, Valid loss: {np.mean(valid_loss)},  valid WER: {wer_score} , valid_cer: {cer_score}\n")
            log_file.close()
            
                    # save best model based on WER
            if (np.mean(train_loss) < best_loss):
                logger.info("saving best model into save checkpoint")
                global best_epoch
                best_epoch = epoch
                best_loss = np.mean(train_loss)
                save_model(os.path.join(hparams['output_folder'],str(hparams.seed),'save','model.ckp'),diffusion_model)
    

        #  sample some texts 

        sampled_latents = diffusion.sample(diffusion_model, sample_size=8,seq_length=latent.shape[1])
        sample_hyp= generate_txt(ae_model,sampled_latents.squeeze(1),tokenizer,hparams,device,input_ids=None)
        sample_text= tokenizer.batch_decode(sample_hyp,skip_special_tokens=True)
        with open(os.path.join(hparams['output_folder'],str(hparams.seed),"samples", f"{epoch}.txt"), 'w') as fp:
          fp.write('\n'.join(sample_text))
        
    
    
    test_loss=[]
    references=[]
    hypothesises=[]
    logger.info("Loading best model from save checkpoint")
    diffusion_model = load_model(os.path.join(hparams['output_folder'],str(seed),'save','model.ckp'),diffusion_model)
    diffusion_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            input_ids, input_ids_lens = batch['input_ids']
            labels, labels_lens = batch['labels']
            batch_size, seq_len= input_ids.shape
            input_ids,labels= input_ids.to(device),labels.to(device)
            latent=generate_latent(ae_model,input_ids, input_ids_lens, tokenizer, hparams,device)
            # sample timestep and generate noisy latents
            t = diffusion.sample_timesteps(batch_size).to(device)
            x_t, noise = diffusion.noise_embedding(latent, t)
            # pass data to diffusion model to predict the noise
            predicted_noise = diffusion_model(x_t, t)

            # Calculate the loss
            loss = mse(noise, predicted_noise)
            denoised_latent = diffusion.sample(diffusion_model, x=latent)
            valid_loss.append( loss.item())

            hyp= generate_txt(ae_model,denoised_latent.squeeze(1),tokenizer,hparams,device,input_ids=input_ids)
            hypothesis= tokenizer.batch_decode(hyp,skip_special_tokens=True)
            reference= tokenizer.batch_decode(input_ids,skip_special_tokens=True)
            hypothesises.extend(hypothesis)
            references.extend(reference)


        wer_score = wer(references,hypothesises)*100
        cer_score= cer(references,hypothesises)*100
        logger.info("Test Epoch %i,Loss %9.4f , Test WER  %9.4f, Test CER %9.4f" % (best_epoch, np.mean(test_loss),wer_score,cer_score))



def generate_latent(model,input_ids,input_ids_lens, tokenizer, hparams,device):
    input_ids = input_ids.to(device)
    
            
    # get latent representation from pretrained ae
    if hparams['ae_model_type'].lower() == 'rnn':
        latent_mu, latent_logvar = model.encoder(input_ids, input_ids_lens)

        
    elif hparams['ae_model_type'].lower() == 'transformer':
        src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
        latent_mu, latent_logvar = model.encoder(input_ids,src_pad_mask)

    elif hparams['ae_model_type'].lower() == 'transformer_with_attention':
        src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
        latent_mu, latent_logvar = model.encoder(input_ids,src_pad_mask)
    
    latent = model.latent_sample(latent_mu, latent_logvar)
     #  add channel to the latent when having 2d latent, since we need to use UNET-1D
    if len(latent.shape) ==2:
        latent= latent.unsqueeze(1)
   
    return latent

def generate_txt(model,latent,tokenizer,hparams,device,input_ids=None):
    if hparams['ae_model_type'].lower() == 'rnn':
        hyp,_ = model.decoder.inference(max_sequence_length=hparams['max_sequence_length'], z=latent,temp=hparams['temp'],mode=hparams['decoder_search'])
    elif hparams['ae_model_type'].lower() == 'transformer':
        hyp,_ = model.decoder.inference(max_sequence_length=hparams['max_sequence_length'], z=latent,temp=hparams['temp'],mode=hparams['decoder_search'])
    elif hparams['ae_model_type'].lower() == 'transformer_with_attention':
        if input_ids is None:
            src_pad_mask = (torch.ones((latent.shape[0], latent.shape[1]), dtype=torch.bool)).to(device)
        else:
            src_pad_mask= generate_pad_mask(input_ids, tokenizer.pad_token_id)
        hyp,_ = model.decoder.inference(max_sequence_length=hparams['max_sequence_length'],memory_key_padding_mask=src_pad_mask,  z=latent,temp=hparams['temp'],mode=hparams['decoder_search'])
    return hyp
        



            



if __name__ == "__main__":
    arg_list = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run Diifussion for text generation Experiment")
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

    # args = parser.parse_args()
    run_opts, overrides = parser.parse_known_args(arg_list)
    run(
        run_opts.params, run_opts.overfitting, run_opts.device, overrides)

