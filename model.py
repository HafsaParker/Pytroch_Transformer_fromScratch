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
    
#STEP1: We will make positional embeddings        

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



    
    
        
        