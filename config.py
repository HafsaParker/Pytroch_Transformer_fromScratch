#get config
#fpath we will save of the model
from pathlib import Path
def get_config():
    return {
    "batch_size":8,
    "num_epochs": 20,
    "lr": 10**-4,
    "seq_len":350,
    "lang_src": 512,
    "lang_tgt": "it",
    "model_folder":"weights",
    "model_basename": "tmodel_",
    "preload":None,
    "tokenizer_file":"tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"


}
#helps to find the path where to save the weightss
def get_weights_file_path(config,_epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{_epoch}.pt"
    return str(Path('.'/model_folder/model_filename))
     
