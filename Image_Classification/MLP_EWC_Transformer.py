import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len = 5,embed_model_dim = 2):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=2, n_heads=1):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d

        #key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim)

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512

        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder

        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.size(1)

        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)

        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)


        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)

        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)

        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat) #(32,10,512) -> (32,10,512)

        return output
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor, n_heads):
        super(EncoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):

        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block

        """

        attention_out = self.attention(key,query,value)  #32x10x512
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out = (self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = (self.norm2(feed_fwd_residual_out)) #32x10x512

        return norm2_out



class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention

    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, embed_dim, num_encoders, expansion_factor, n_heads):
        super(TransformerEncoder, self).__init__()

        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([EncoderBlock(embed_dim, expansion_factor, n_heads) for i in range(num_encoders)])

    def forward(self, x):
        # x = self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x,x,x)

        return x  #32x10x512
    
class MLPTransformer(nn.Module):
    def __init__(self, seq_length, embed_dim, num_encoders, expansion_factor, n_heads):
        super(MLPTransformer, self).__init__()

        """
        Args:
           embed_dim:  dimension of embedding
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """

        self.encoder = TransformerEncoder(seq_len=seq_length, embed_dim=embed_dim, num_encoders=num_encoders, expansion_factor = expansion_factor, n_heads = n_heads)
        self.mlp = nn.Sequential(
                        # nn.Linear(seq_length * embed_dim,100),
                        # nn.ReLU(),
                        # nn.Linear(100,10)
                        nn.Linear(seq_length * embed_dim,10)

                                )

    def forward(self, x):
        """
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """

        out = self.encoder(x)
        out = out.reshape(out.shape[0], -1)
        out = self.mlp(out)

        return out
    
def compute_fisher_matrix(model, criterion, batch_history):
        optpar_dict = {}
        fisher_dict = {}
        model_dict = dict(model.named_parameters())

        trainable_parameter_names = []
        trainnable_parameters = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_parameter_names.append(name)
                trainnable_parameters.append(param)
                # Initialize dictionaries with all trainable parameters
                optpar_dict[name] = torch.zeros_like(param.data)
                fisher_dict[name] = torch.zeros_like(param.data)

        fisher_mem = len(batch_history) #// 2 if len(batch_history) > 0 else 1 
        for item_num, (images, labels) in enumerate(batch_history[::-1]):
            if item_num < fisher_mem:
                outputs = model(images)
                loss = criterion(outputs, labels)
                model.zero_grad() # Clear gradients before computing new ones
                loss.backward()

                for name in trainable_parameter_names:
                    if model_dict[name].grad is not None: # Check if gradient exists
                        optpar_dict[name] += model_dict[name].data.clone()
                        fisher_dict[name] += model_dict[name].grad.data.clone().pow(2)

        # Only divide if batch_history is not empty
        if len(batch_history) > 0:
            for name in trainable_parameter_names:
                optpar_dict[name] /= len(batch_history)
                fisher_dict[name] /= len(batch_history)


        return fisher_dict, optpar_dict

def train(epochs, bc, shape, batch_history, args, model, device, task):
  model.to(device)
  model.train()
  opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
  criterion = nn.CrossEntropyLoss()
  l = []
  lc = 0
  #EWC
  trainable_parameter_names = []
  trainnable_parameters = []

  for name, param in model.named_parameters():
      if param.requires_grad:
          trainable_parameter_names.append(name)
          trainnable_parameters.append(param)

  fisher_dict, optpar_dict = compute_fisher_matrix(model, criterion, batch_history)
  #EWC
  for _ in range(epochs):
      running_loss = 0
      for i, (images, labels) in enumerate(task):
        if i < bc:
          images, labels = images.to(device), labels.to(device)
          images = images.reshape(-1,shape,shape)

          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)
          lc += 1
          #EWC
          if len(batch_history) > 0 and args.ewc:
            print("EWC")
            for n, p in zip(trainable_parameter_names, trainnable_parameters):
                ewc_regularizer = args.ewc_lambda * torch.sum(fisher_dict[n] * (p - optpar_dict[n]) ** 2)
                loss += ewc_regularizer
          #EWC
          # Backward pass
          opt.zero_grad()
          loss.backward()
          opt.step()

          running_loss += loss.item()

      epoch_loss = running_loss / lc
      l.append(epoch_loss)

  for i, (images, labels) in enumerate(task):
    if i < bc:
        images, labels = images.to(device), labels.to(device)
        images = images.reshape(-1,shape,shape)
        batch_history.append((images.to(device), labels.to(device)))

def evaluate(bc, shape, model, device, task):
  model.eval()
  criterion = nn.CrossEntropyLoss()
  running_loss = 0
  correct = 0
  c = 0
  lc = 0
  for i, (images, labels) in enumerate(task):
    if i < bc:
      images, labels = images.to(device), labels.to(device)
      images = images.reshape(-1,shape,shape)

      # Forward pass
      outputs = model(images)

      loss = criterion(outputs, labels)
      _, predicted = torch.max(outputs, 1)

      correct += (predicted == labels).sum().item()
      c += len(predicted)
      lc += 1


      running_loss += loss.item()
  epoch_loss = running_loss / lc
  return(correct/c)