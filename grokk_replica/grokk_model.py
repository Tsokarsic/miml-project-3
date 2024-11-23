import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer,lstmnetwork,mlpnetwork
from utils import causal_attn_mask, parameter_norm

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
        param_norm = parameter_norm(self)
        return loss, {'loss': (loss.item(), x.shape[0]), 'accuracy': (accuracy.item(), x.shape[0]),
                      # 'attn_entropy': (attn_entropies, len(attns)*x.shape[0]*(x.shape[1]-1)),
        'param_norm': (param_norm, 1)}