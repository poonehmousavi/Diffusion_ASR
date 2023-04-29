#!/bin/bash
module load python/3.10.2
source $HOME/myenv/bin/activate

scp $HOME/projects/def-ravanelm/datasets/librispeech/train-clean-360.tar.gz $SLURM_TMPDIR
scp $HOME/projects/def-ravanelm/datasets/librispeech/test-clean.tar.gz $SLURM_TMPDIR
scp $HOME/projects/def-ravanelm/datasets/librispeech/dev-clean.tar.gz $SLURM_TMPDIR


cd $SLURM_TMPDIR
tar -zxf train-clean-360.tar.gz
tar -zxf test-clean.tar.gz
tar -zxf dev-clean.tar.gz

MODEL=$1


cd $HOME/Diffusion_ASR/data_utils/
python librispeech_prepare.py $SLURM_TMPDIR/LibriSpeech --save_folder=$SLURM_TMPDIR/LibriSpeech

python  train_text_vae.py hparams/train_rnn_vae.yaml  --hub_cache_dir=$SLURM_TMPDIR/cache_dir/ --output_folder=$HOME/scratch/Difussion_ASR/rnn_vae/  --train_csv=train-clean-360.csv --data_folder=$SLURM_TMPDIR/LibriSpeech/ --optimizer=adam --train_batch_size=256 --valid_batch_size=128 --test_batch_size=128 --number_of_epochs=20 
