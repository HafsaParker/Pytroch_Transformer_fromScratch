import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from datasets import load_datasets 
from model import build_transformer 
from tokenizers import Tokenizer
from config import get_weights_file_path,get_config
#tokenizer we need
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace #split the words via white space
from pathlib import Path
from dataset import BilingualDataSet, casual_mask
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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


##MODEL LOADING

def get_model(config,vocab_src_lang,vocab_tgt_lang):
    model = build_transformer(vocab_src_lang,vocab_tgt_lang,config['seq_len'],config['seq_len'],config['d_model'])
    return model

#traing loop
def train_model(config):
    #define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device",device)
    Path(config['model_folder']).mkdir(parents =True,exist_ok=True)
    #laod data set
    train_data_loaders,val_dataloader,tokenizer_src,tokenizer_tgt = get_dataset(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    #tensorboard
    writer= SummaryWriter(config["experiment_name"])
    #create optimizer
    optimizer =  torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)
    #restoring the modela nd optimizer comfig on crash
    initital_epoch = 0
    global_step =0
    if config["preload"]:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f'Preloading model{model_filename}')
        state = torch.load(model_filename)
        initital_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    for epoch in range(initital_epoch,config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_data_loaders,desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            #Run the tensors through the transformer
            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output = model.project(decoder_output)
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({f'loss':f'{loss.item():6.3f}'})
            #log the loss
            writer.add_scalar('train loss',loss.item(),global_step)
            writer.flush()
            #backpropagate the loss
            loss.backward()
            #update the weights
            optimizer.step()
            optimizer.zero_grad()
            global_step +=1
        #save the model at the end pf every epoch\
        model_filename = get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            "epoch":epoch,
            "model_state_dict":optimizer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "global_step": global_step


        },model_filename)

if __name__ == '__main__':
    config = get_config()
    train_model(config)








        
             









    




