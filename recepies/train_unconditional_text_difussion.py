import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging

from data_utils.tokenizer import get_tokenizer
from data_utils.librispeech_dataset import LibriSpeechDataset
from utils.helper import *
from models.VAE import RNNVariationalAutoencoder, TransformerVariationalAutoencoder,AttentionTransformerVariationalAutoencoder
from models.transformer import generate_pad_mask,generate_square_subsequent_mask,Transformer_Diffusion
from difussion.ddpm import Diffusion

overfitting_number= 10

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
 

def run(params_file,data_root, overfitting_test,device):

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
    
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # append_root_folder to the data, cache and output folder
    hparams['output_folder']=  os.path.join(data_root,hparams['output_folder'])
    hparams['data_folder']=  os.path.join(data_root,hparams['data_folder'])
    hparams['hub_cache_dir']=  os.path.join(data_root,hparams['hub_cache_dir'])


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

    if hparams['diff_model_type'].lower() == 'transformer':
        diffusion_model = Transformer_Diffusion(**hparams['diffusion_model'])
    elif hparams['diff_model_type'].lower() == 'unet':
        diffusion_model = Transformer_Diffusion(**hparams['diffusion_model'])
    else:
        logger.error(f"{hparams['diff_model_type']} is not among supperted model types for diffusion. Supported models are  transformer and unet")


    # Define Optimizer
    if hparams['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(diffusion_model.parameters(), lr=float(hparams['lr']))
    elif hparams['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=float(hparams['lr']))
    else:
        logger.error(f"{hparams['optimizer']} is not among supperted optimizer. Supported models are SGD and Adam")    
    
    mse = nn.MSELoss()
    diffusion = Diffusion(**hparams['diffusion'], device=device)


    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        loss=[]

        for i, batch in enumerate(tqdm(train_loader)):
            input_ids, input_ids_lens = batch['input_ids']
            dec_input_ids, dec_input_ids_lens = batch['dec_input_ids']
            labels, labels_lens = batch['labels']
            input_ids,dec_input_ids,labels= input_ids.to(device),dec_input_ids.to(device),labels.to(device)
            batch_size, seq_len= input_ids.shape
            
            # get latent representation from pretrained ae
            latents =""
            
            # sample timestep and generate noisy latents
            t = diffusion.sample_timesteps(batch_size).to(device)
            x_t, noise = diffusion.noise_embedding(latents, t)
            
            # pass data to diffusion model to predict the noise
            predicted_noise = diffusion_model(x_t, t)

            # Calculate the loss
            loss = mse(noise, predicted_noise)

            # calculate gradient and update parameter 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss.add( loss.item())

        #  sample some texts 
        sampled_images = diffusion.sample(diffusion_model, n=batch_size)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))


        # save model 
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Text generation Experiment")
    parser.add_argument(
        "params",
        help='path to params file',
    )
    parser.add_argument(
        "--root_folder",
        help='root_folder',
        required=True
    )
    parser.add_argument(
        "--device",
        help='device',
        default='cuda'
    )
    parser.add_argument("--overfitting",action='store_true', help="in overfitting test mode",default=False)

    args = parser.parse_args()
    run(
        args.params, args.root_folder, args.overfitting, args.device)
