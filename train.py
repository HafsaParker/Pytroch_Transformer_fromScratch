import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from datasets import load_datasets  
from tokenizers import Tokenizer
#tokenizer we need
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace #split the words via white space
from pathlib import Path
from dataset import BilingualDataSet, casual_mask

def get_all_sentence(ds, lang):
    """
    this func will iterate through the sentences and work 
    on the sentences of th language we will choose. 
    

    Each item in the sentence is the pair of ['eng','italian']
    """
    for item in ds:
        yield item['translation'][lang]



# Method that build tokenizer
def get_or_build_tokenizer(config, ds, lang  ):
    """
    config  = config of the model
    ds = dataset
    lang = lang for which we will build the tokenizer
    
    """
    #path of the tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    #if tokenizer donot exists we create it
    if not Path.exists(tokenizer_path):
        #Unk_token means that if the model doesnt recongnise the word
        #it will replace it with tokenier
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace
        trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency =2 )
        #train the tokenizer
        tokenizer.train_from_iterator(get_all_sentence(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
         
#code to load the dataset

def get_dataset(config):
    ds_raw = load_datasets('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split = 'train')
    #build the toeknizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    #train and valid split
    train_ds_size = int(0,9*len(ds_raw))
    val_ds_size = len(ds_raw)-train_ds_size
    train_ds_size,val_ds_size = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataSet(train_ds,tokenizer_src,tokenizer_tgt,config['lang_src'],config["lang_tgt"],config["seq_len"])
    val_ds = BilingualDataSet(val_ds_size,tokenizer_src,tokenizer_tgt,config['lang_src'],config["lang_tgt"],config["seq_len"])

    #checking max seq len of source and target for train and val.
    max_src_len = 0
    max_tgt_len = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
    print(f"Max lenght of source sentence: {max_src_len}")
    print(f"Max lenght of target sentence: {max_tgt_len}")

    #data loaders
    train_data_loaders =DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)
    return train_data_loaders,val_dataloader,tokenizer_src,tokenizer_tgt




    




