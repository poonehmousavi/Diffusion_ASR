# Stable Diffusion for Unconditional Text Generation

The objective of Text Generation is to produce written content that is virtually identical to humangenerated
text. This process is commonly referred to as ”natural language generation” in academic
literature. The ability to automatically generate
coherent and semantically meaningful text has
numerous practical applications, including machine
translation, dialogue systems, and image
captioning. Traditional causal language models,
such as RNN-LM and GPT, generate sentences
one word at a time, without considering an overall
sentence representation. This approach could
be problematic because errors in the initial steps
could propagate throughout the entire prediction.
Recent studies have shown that diffusion models
exhibit promising results in image generation. In
this work, I will investigate the use of the stable
diffusion model to address the challenges of text
generation tasks.

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

### Using Conda (recommended)

First of all, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Clone the repository, navigate to `<path-to-repository, open a terminal and run:

```
conda create -n text_diffusion_env
```

Activate yot env:

 ```
source activate text_diffusion_env
```
To deactivate it, run:

```
conda deactivate
```

### Using Pip

Project dependencies (pinned to a specific version to reduce compatibility and reproducibility issues)
To install dependencies, run: 

```
pip install -r  requiremnet.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart
### Data Preparation
You can download LibriSpeech beforehand, and store them for later use.
To do so, navigate to `<path-to-repository>/data_utils`, open a terminal and run:

```
python librispeech_prepare.py  <data_folder> --save_folder=<output_folder>
```

### Traing VAE

Navigate to `<path-to-repository>/recipes/`, open a terminal and run
(remember to activate the virtual environment via `source activate text_diffusion_env` if you installed the project using Conda):

```
python train_text_vae.py hparams/train_<vae-method>_vae.yaml
```

**NOTE**: You could change the hyper-param in yaml file or you could pass it as command input and it will override values in yaml file. For example, to override the number of epoch, you coudl run, 
```
python train_text_vae.py hparams/train_<vae-method>.yaml --number_of_epochs=1
```

### Traing Difdusion

Navigate to `<path-to-repository>/recipes/`, open a terminal and run
(remember to activate the virtual environment via `source activate text_diffusion_env` if you installed the project using Conda):

```
python train_text_vae.py hparams/train_<vae-method>.yaml
```

**NOTE**: You could change the hyper-param in yaml file or you could pass it as command input and it will override values in yaml file. For example, to override the number of epoch, you coudl run, 
```
python train_unconditional_text_difussion.py hparams/train_<vae-method>_diffusion.yaml --number_of_epochs=1
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[mousavi.pooneh@gmail.com](mousavi.pooneh@gmail.com)
