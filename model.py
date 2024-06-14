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
        def forward(self,q,k,v,mask):
            """
            mask =  if we dont want to somewords to interact with other 
            words we put their value to a small number.
            """
            


        
              
         
    
    
        
        