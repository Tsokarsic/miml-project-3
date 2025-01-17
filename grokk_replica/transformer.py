from unicodedata import bidirectional

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Nonlinearity(nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()
    def forward(self, x):
        # 使用torch.where来选择不同的激活方式
        # 当x > 1时，使用x^2；否则使用x（线性）
        return torch.where(x > 1, x ** 2, x)
class mlpblock(nn.Module):
    def __init__(self,input_dim,output_dim,add_bias,norm_method):
        super(mlpblock, self).__init__()
        self.ff=nn.Linear(input_dim,output_dim,bias=add_bias)
        self.batchnorm=nn.BatchNorm1d(output_dim)
        self.layernorm=nn.LayerNorm(output_dim)
        self.norm=norm_method
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.05)
    def forward(self,x):
        x=self.ff(x)
        if self.norm=='batchnorm':
            x=self.batchnorm(x)
        if self.norm=='layernorm':
            x=self.layernorm(x)
        # x = torch.where(x > 1, x ** 2, F.relu(x))
        # x=self.leakyrelu(x)
        x=F.relu(x)
        return x

class mlpnetwork(nn.Module):
    def __init__(self,vocab_size,num_layers,intermediate_dim,output_size,add_bias,norm_method,num_p):
        super(mlpnetwork, self).__init__()
        self.hidden_dim=vocab_size
        # self.embeddings=nn.Embedding(vocab_size,self.hidden_dim)
        self.mlpblock1=mlpblock(2*num_p*self.hidden_dim,intermediate_dim,add_bias,norm_method)
        self.mlpblock2=nn.Sequential(*[mlpblock(intermediate_dim,intermediate_dim,add_bias,norm_method) for _ in range(num_layers-1)])
        self.output_layer=nn.Linear(intermediate_dim,output_size,bias=add_bias)
    def forward(self,x):
        #x=self.embeddings(x)
        x=F.one_hot(x,self.hidden_dim).to(torch.float32)
        x=x.reshape(x.shape[0],-1)
        x=self.output_layer(self.mlpblock2(self.mlpblock1(x)))
        return x

    def xavier_init(model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def kaiming_init(model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

class lstmnetwork(nn.Module):
    def __init__(self, vocab_size, intermediate_dim, num_layers, dropout, bidirectional,output_size):
        super(lstmnetwork, self).__init__()
        self.hidden_dim = vocab_size
        self.lstm=nn.LSTM(self.hidden_dim, intermediate_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.ff1=nn.Linear(intermediate_dim,output_size)
        self.ff2=nn.Linear(2*intermediate_dim,output_size)
        self.embeddings=nn.Embedding(vocab_size,self.hidden_dim)
        self.bidirectional=bidirectional

    def forward(self,x):
        x=F.one_hot(x,self.hidden_dim).to(torch.float32)
        output,_=self.lstm(x)
        if self.bidirectional:
            pred=self.ff2(output)
        else:
            pred=self.ff1(output)
        return pred

    def xavier_init(model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def kaiming_init(model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))



class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_dim, attn_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.attn_dim = attn_dim
        self.dropout = nn.Dropout(p=dropout)
        self.key_proj = nn.Linear(hidden_dim, self.attn_dim*self.heads)
        self.val_proj = nn.Linear(hidden_dim, self.attn_dim*self.heads)
        self.query_proj = nn.Linear(hidden_dim, self.attn_dim*self.heads)
        self.output_proj = nn.Linear(self.attn_dim*self.heads, hidden_dim)

    def forward(self, queries, keys, values, mask, past_kv=None):
        assert keys.shape[1] == values.shape[1], 'keys and values time dimension must match'
        assert past_kv is None or past_kv[0].shape[1] == past_kv[1].shape[1], 'cached keys and values time dimension must match'
        # queries/keys/values = (batch, time, hidden_dim)
        # mask = (batch, query_time, key_time) - bool tensor, True if should mask
        # past_kv = tuple of (past_k=(batch, time, head, hidden_dim), past_v=(batch, time, head, hidden_dim)) or None
        # returns:
        # attn_matrix = (batch, head, query_time, key_time)
        # attn_output = (batch, query_time, hidden_dim)
        # tuple of updated past_kv

        batch, time, _ = queries.shape
        key_heads = self.key_proj(keys).reshape(batch, time, self.heads, self.attn_dim)
        val_heads = self.val_proj(values).reshape(batch, time, self.heads, self.attn_dim)
        query_heads = self.query_proj(values).reshape(batch, time, self.heads, self.attn_dim)
        if past_kv is not None:
            past_k, past_v = past_kv
            key_heads = torch.cat([past_k, key_heads], dim=1)
            val_heads = torch.cat([past_v, val_heads], dim=1)
        attn_matrix = F.softmax((torch.einsum('bqhd,bkhd->hbqk', query_heads, key_heads)
                                 / math.sqrt(self.attn_dim)).masked_fill(mask, float('-inf')), dim=-1)
        attn_matrix = self.dropout(attn_matrix.transpose(0, 1).contiguous())
        combined_vals = torch.einsum('bkhd,bhqk->bqhd', val_heads, attn_matrix).reshape(batch, time, self.attn_dim*self.heads)
        attn_output = self.output_proj(combined_vals)
        return attn_output, attn_matrix, (key_heads, val_heads)

class TransformerBlock(nn.Module):
    def __init__(self, heads, hidden_dim, attn_dim, intermediate_dim, dropout=0.1, pre_norm=True):
        super(TransformerBlock, self).__init__()
        self.pre_norm = pre_norm
        self.attn = MultiHeadAttention(heads, hidden_dim, attn_dim, dropout=dropout)
        self.ff1 = nn.Linear(hidden_dim, intermediate_dim)
        self.ff2 = nn.Linear(intermediate_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)


    def forward(self, x, attn_mask, past_kv=None):
        if not self.pre_norm:
            attn_output, attn_matrix, past_kv = self.attn(x, x, x, attn_mask, past_kv=past_kv)
            x = self.layer_norm1(self.dropout1(attn_output) + x)
            mlp_out = self.ff2(self.dropout2(F.gelu(self.ff1(x))))
            x = self.layer_norm2(self.dropout3(mlp_out) + x)
        else:
            x_norm1 = self.layer_norm1(x)
            attn_output, attn_matrix, past_kv = self.attn(x_norm1, x_norm1, x_norm1, attn_mask, past_kv=past_kv)
            x = self.dropout1(attn_output) + x
            x_norm2 = self.layer_norm2(x)
            mlp_out = self.ff2(self.dropout2(F.gelu(self.ff1(x_norm2))))
            x = self.dropout3(mlp_out) + x
        return x, attn_matrix, past_kv

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_length, heads, attn_dim, intermediate_dim, num_blocks, block_repeats, output_size, dropout=0.1, pre_norm=True):
        super(Transformer, self).__init__()
        self.pre_norm = pre_norm
        self.hidden_dim = vocab_size
        self.block_repeats = block_repeats
        self.max_length = max_length
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(heads, self.hidden_dim, attn_dim, intermediate_dim, dropout=dropout, pre_norm=pre_norm) for _ in range(num_blocks)
        ])
        self.embeddings = nn.Embedding(vocab_size, self.hidden_dim)
        self.positions = nn.Embedding(max_length, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, output_size)
        self.dropout = nn.Dropout(p=dropout)
        if self.pre_norm:
            self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, attn_mask, past_kvs=None):
        # x = (batch, time)
        # attn_mask = (batch, query_time, key_time)
        # past_kvs = list of past_kvs for each layer
        # print(x.shape)
        attns = []
        new_past_kvs = []
        initial_pos = 0
        if past_kvs is not None:
            initial_pos = past_kvs[0][0].shape[1]
        assert initial_pos+x.shape[1] <= self.max_length, 'sequence too long'
        # x = self.dropout(
        #     self.embeddings(x) * math.sqrt(self.hidden_dim)
        #     + self.positions.weight[initial_pos:initial_pos + x.shape[1], :]
        # )
        x = self.dropout(F.one_hot(x,self.hidden_dim).to(torch.float32) * math.sqrt(self.hidden_dim) + self.positions.weight[initial_pos:initial_pos+x.shape[1], :])
        # # #print(x.shape)
        step = 0
        for _ in range(self.block_repeats):
            for i in range(len(self.transformer_blocks)):
                x, attn, past_kv = self.transformer_blocks[i](x, attn_mask, past_kv=past_kvs[step] if past_kvs is not None else None)
                attns.append(attn)
                new_past_kvs.append(past_kv)
                step += 1
        if self.pre_norm:
            x = self.norm(x)
        #print(self.output(x).shape)
        return self.output(x), attns, new_past_kvs

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def kaiming_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

