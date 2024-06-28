import torch
from torch import nn
import math
#STEP1: We will make input embeddings
 
class InputEmbeddings(nn.Module):
    """dimension of the vector
       vocab size --> how many words are there in the vocab"""
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model) #nn.embedding is just a layer
    def forward(self,x):
        """
        we will let pytorch to map the embedding
        """
        return self.embedding(x) * math.sqrt(self.d_model) #its in paper
    
#STEP2: We will make positional embeddings        

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,seq_len,dropout):
      """
      seq_len = max lenght of the sentence.
      dropout = to make model less over fit
      """
      super().__init()
      self.d_model = d_model
      self.seq_len = seq_len
      self.dropout = dropout

      #create a matrix of shape (seq_len x d_model)
      pe = torch.zeros(seq_len,d_model)
      #create a vector of shape(seq_len,1)
      position = torch.arrange(0,seq_len-1,dtype = float).unsqueeze(1)
      div_term = torch.exp(torch.arrange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
      #sine --> even positions
      #cosine --> odd positions
      pe[:,0::2] = torch.sin(position*div_term)
      pe[:,1::2] = torch.cos(position*div_term)
      
    #Add batch dimenssion to the tensor to apply it to the full sentence.
      pe = pe.unsqueeze(0)
      self.register_buffer('pe',pe)
    def forward(self, x):
        """
        Add positional encod to every word in the sentence
        requires_grad_(False) = this will tell the model that it will be 
        calculated only once.
        Dropout --> is a technique used in neural networks to improve model
        performance and reduce overfitting
        """
        x = x+(self.pe[:,x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

#STEP3 LayerNormalization
class LayerNormalization(nn.Module):
    """
    Just need 1 parameter
    eps = very small number that you need to give to the model
    we need this because its in the formula for layer norm.
    nn. Parameter --> is used to explicitly specify which tensors should 
    be treated as the model's learnable parameters
    """
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.aplha = nn.Parameter(torch.ones(1)) #will be multiplied
        self.bias = nn.Parameter(torch.zeros(0)) # added
    def forward(self,x):
        mean = x.mean(dim = -1 , keepdim = True)
        std = x.std(dim = -1 , keepdim = True)
        return self.aplha*(x-mean)/(std +self.eps)+self.bias
#STEP 4 FEED FORWARD LAYER
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) #w1 and b1
        self.dropout = dropout
        self.linear_2 = nn.Linear(d_ff,d_model) #w2 and b2
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
            
#Multihead Attention
class MultiheadAttentionblock(nn.Module):
    """
    In multihead attention we have a input (seq,d_model) we covert
    it into 3 matroces Q K V its same as input and multiply by
    Wq Wk Wv respectively.
    (Q K V) x (Wq Wk Wv) = (Q' K' V') -split into--> Number of heads X Wo =MH-A(same seq as input)
    self.h = number of heads
    we have to divide the d_model with h thus d_model and h value should
    be that they are divisible.
    """
    def __init__(self,d_model,h,dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d model is not divisible by h"
        self.d_k = d_model//h # as in paper
        #Now defining the matrices for multiplication
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_0 = nn.Linear(d_model,d_model)
    # Now we have to calculate the attention
    @staticmethod
    def attention(query,key,value,mask,dropout,nn.Dropout):
        d_k  = query.shape[-1]
        #now we will apply the formula
        """
        @ --> matrix multiplication in pytorch
        before applying sofmax we hae to apply mask
        """
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k) #last 2 dimenssion
        if mask is not None:
            attention_scores.masked_fill(mask == 0,-1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
        

    def forward(self,q,k,v,mask):
        """
        mask =  if we dont want to somewords to interact with other 
        words we put their value to a small number.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        
        x, self.attention_scores= MultiheadAttentionblock.attention(query,key,value,mask,self.dropout)
         #
        x= x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        #AS in the paper
        return self.w_0(x)

#Residual Connection the arrows in the encoder layers 
class ResidualConnection(nn.Module):
    """
    connection is between add and the norm and the previus layer

    sublayer --> next layer
    """
    def __init__(self,dropout):
        super().__init__()
        self.dropout = dropout 
        self.norm = LayerNormalization(dropout)
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
    
#we will now create a Encoder block that contains the 1 multihead attention
# 2 Add and norms and 1 feed forward.
class EncoderBlock(nn.Module):

    """
    src_mask --> the mask that we want to apply to the unput of theencoder
    bcz we want to hide the interaction of the padding word with the otehr word.

    """
    def __init__(self, self_attention: MultiheadAttentionblock, feed_forward_block:FeedForwardBlock,dropout:float) -> None:
    
        super().__init__() 
        self.self_attention = self_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout)for _ in range(2)])  
    def forward(self,x,src_mask):
        #step 1 = x to multihead and ad and norms and combine the 2
        #other x is comming from selfattention
        #xxx is query key and value
        #here wwe are calling forward func of the multihead attention
        x = self.residual_connections[0](x,lambda x: self.self_attention(x,x,x,src_mask))
        #step2 = feed forward
        #this means the x value of upperlayer can be added to next layer.
        x= self.residual_connections[1](x, self.feed_forward_block)
        return x

    
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self,x,mask):
        for layers in self.layers:
            x= layers(x,mask)
        return self.norm(x)

"""
DECODER

Output embedding == input embedding
and also positional encoding 
decoder block is mad eup of 3 sublayers

the first part of decoder is self attention as KVQ is comming from output 
embedding 
however, in the sec part k and v are comming from the encoder block
while q is comming fromt the decoder block. (Cross attention)
cross_attention_block:MultiheadAttentionblock -> its same but we will give it diff params
"""    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiheadAttentionblock, cross_attention_block:MultiheadAttentionblock, feed_forward_block: FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout)for _ in range(3)])
    """
    x= input of the decoder 
    src_mask = mask applied to the encoder
    targetmask  = mask applied to the decoder
    src and target mask bcz we have a source lang = eng and
    trgtmask = italian
    """
    def forward(self, x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connections[0](x,lambda x: self.self_attention(x,x,x,tgt_mask))
        #cross attention
        x= self.residual_connections[1](x, self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=  self.residual_connections[2](x, self.feed_forward_block)
        return x
class decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x= layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    
# Linear layer

class Projectionlayer(nn.Module):
    """
    project embeding into the words.
    (batch,seq_len,d_model)  --> (Batch,seq_len,vocab_size)
    
    """
    def __init__(self, d_model,vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.proj(x),din = -1)
class Tranformer(nn.Module):
    def __init__(self, enoder: Encoder,decoder: decoder, src_embed: InputEmbeddings,tgt_embed: InputEmbeddings,src_pos: PositionalEmbedding,tgt_pos: PositionalEmbedding,projection_layer:Projectionlayer) -> None:

        super().__init__()
        self.encoder = enoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        #first apply target embedd to target sentence
        tgt = self.tgt_embed(tgt)
        #pos encode to tgt sen
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    def project(self,x):
        return self.projection_layer(x)

#NOW WE HAVE TO COMBINE ALL THE BLOCKS TOGETHER
def build_transformer(src_vocab_size,tgt_vocab_size,src_seq_len,tgt_seq_len,d_model=512,N:int = 6,h=8,dropout =0.1,d_ff = 2048) -> Tranformer:   
    #Create e,bdedding ayer
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)
    #Create positional encoding layer
    src_pos = PositionalEmbedding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEmbedding(d_model,tgt_seq_len,dropout)
    #Create the encoder blocks
    encoder_blocks =[]
    for _ in range(N):
        encoder_self_attention_block =MultiheadAttentionblock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block  = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    #create decoder layer
    decoder_blocks =[]
    for _ in range(N):
        decoder_self_attention_block =MultiheadAttentionblock(d_model,h,dropout)
        decoder_cross_attention_block = MultiheadAttentionblock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block  = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    #create the encoder and the decoder
    encoder = Encoder(nn.Module(encoder_blocks))
    decoder = decoder(nn.Module(decoder_blocks))
    #Create projection layer
    Projection_layer = Projectionlayer(d_model,tgt_vocab_size)
    #create the transformer
    transformer = Tranformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,Projection_layer)

    #INITIALIZE THE PARAMS
    for p in transformer.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)
    return transformer










