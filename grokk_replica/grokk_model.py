import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer,lstmnetwork,mlpnetwork
from utils import causal_attn_mask, parameter_norm
import math

class GrokkModel(nn.Module):
    def __init__(self, transformer_config,lstm_config ,mlp_config,vocab_size, output_size, device, mode):
        super(GrokkModel, self).__init__()
        self.mlpnetwork = mlpnetwork(**mlp_config,vocab_size=vocab_size, output_size=output_size)
        self.transformer = Transformer(**transformer_config, vocab_size=vocab_size, output_size=output_size)
        self.lstmnetwork = lstmnetwork(**lstm_config,vocab_size=vocab_size,output_size=output_size)
        self.device = device
        self.mode=mode
    
    def forward(self, x):
        if self.mode=='transformer':
            attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(self.device)
            predictions, _,_ = self.transformer(x, attn_mask)

        if self.mode=='lstm':
            predictions = self.lstmnetwork(x)

        if self.mode == 'mlp':
            predictions = self.mlpnetwork(x)

        return predictions


    def get_loss(self, x, y):
        predictions= self(x)
        if self.mode !='mlp':
            predictions=predictions[:, -1, :]
        # print(torch.argmax(predictions[:, -1, :], dim=-1), x[:, -1])
        loss = F.cross_entropy(predictions, y)
        accuracy = (torch.argmax(predictions, dim=-1) == y).float().mean()
        #attn_entropies = sum([-(attn * torch.log(attn+1e-7)).sum(dim=-1).mean().item() for attn in attns]) / len(attns)
        if self.mode=='lstm':
            param_norm = parameter_norm(self.lstmnetwork)
            max_norm = max(param.data.abs().max().item() for param in self.lstmnetwork.parameters() if param.requires_grad)
        elif self.mode=='transformer':
            param_norm = parameter_norm(self.transformer)
            max_norm = max(param.data.abs().max().item() for param in self.transformer.parameters() if param.requires_grad)
        else:
            param_norm = parameter_norm(self.mlpnetwork)
            max_norm = max(param.data.abs().max().item() for param in self.mlpnetwork.parameters() if param.requires_grad)
        return loss, {'loss': (loss.item(), x.shape[0]), 'accuracy': (accuracy.item(), x.shape[0])}
                      # 'attn_entropy': (attn_entropies, len(attns)*x.shape[0]*(x.shape[1]-1)),

    def get_train_accuracy(self, x, y):
        predictions= self(x)
        if self.mode !='mlp':
            predictions=predictions[:, -1, :]
        # print(torch.argmax(predictions[:, -1, :], dim=-1), x[:, -1])
        loss = F.cross_entropy(predictions, y)
        accuracy = (torch.argmax(predictions, dim=-1) == y).float().mean()
        #attn_entropies = sum([-(attn * torch.log(attn+1e-7)).sum(dim=-1).mean().item() for attn in attns]) / len(attns)
        if self.mode=='lstm':
            param_norm = parameter_norm(self.lstmnetwork)
            max_norm = max(param.data.abs().max().item() for param in self.lstmnetwork.parameters() if param.requires_grad)
        elif self.mode=='transformer':
            param_norm = parameter_norm(self.transformer)
            max_norm = max(param.data.abs().max().item() for param in self.transformer.parameters() if param.requires_grad)
        else:
            param_norm = parameter_norm(self.mlpnetwork)
            max_norm = max(param.data.abs().max().item() for param in self.mlpnetwork.parameters() if param.requires_grad)
        return accuracy


    def xavier_init(model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def kaiming_init(model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def orthogonal_init(self, gain=1):
        """
        使用正交初始化方法初始化模型权重参数
        Args:
            gain (float): 用于缩放的增益因子。默认值为 1。
        """
        for p in self.parameters():
            if p.dim() > 1:  # 仅对多维参数应用正交初始化
                nn.init.orthogonal_(p, gain=gain)
