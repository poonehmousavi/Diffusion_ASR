import torch
import os 

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def create_experiment_directory(experiment_directory):
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)


def save_model(model_file_name,model):
    torch.save(model.state_dict(), model_file_name)


def load_model(model_file_name,model):
    model.load_state_dict(torch.load(model_file_name))
    return model

def resolve_overrides_params(hparam, overrides):
    for item in overrides:
        param,value= parse_override_arg(item)
        if param in hparam.keys():
            hparam[param] =type(hparam[param])(value)
    return hparam
   

def parse_override_arg(argument):
    param, value= argument.split("=")
    param= param.split("--")[1]
    return param, value