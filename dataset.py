import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataSet(Dataset):
    def __init__ (self,ds, tokenizer_src,tokenizer_tgt,src_lan,
                tgt_lan ,seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt= tokenizer_tgt
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.seq_len =seq_len
        #  start of sentence to token ID
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype = torch.int64) 
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype = torch.int64) 
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype = torch.int64) 
    def __len__(self):
        return (self.ds)
    def __getitem__(self, index: Any) -> Any:
        """
       extracting the real pair from HF dataset
       extracting src_text and target_text

        """
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lan]
        tgt_text = src_target_pair['translation'][self.tgt_lan]

        #each text into token into token ID
        # sent --> words --> number into vocab
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #PADDING --> we dont always have same seq lenght in the sentence . 
        #padding helps t fill the gaps.

        #calculating the numbers of padding token needed
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2 #2--> sos nd eos
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 #1 --> sos

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError("Sentence is too long")
        

        #coding tensors for the encoder and decoder input.
        # 1 sent = input_encoder 1 = input_decoder
        # 1 sent  = output decoder ==> label
        #encoder_input = torch.cat(
        #    [
        #        self.sos_token,
        #        torch.tens
#
#
        #    ]
#
#
        #)
#








        








     