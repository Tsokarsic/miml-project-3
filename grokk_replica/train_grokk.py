import torch
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import IterableDataset
from utils import causal_attn_mask, parameter_norm
from datasets import *
from utils import combine_logs
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
from tqdm.auto import tqdm
import hydra
from optimizer import *
from omegaconf import DictConfig, OmegaConf
from typing import Optional,Dict,Literal
from collections import deque

from grokk_replica.datasets import KSumDataset
from load_objs import load_item

class GroupDataset(IterableDataset):
    def __init__(self, dataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

def lr_decay(iter_number,method,rate,warmup_steps):
    def f(s):
        if s<warmup_steps :
            return s/warmup_steps
        else:
            s=s-warmup_steps
            iterstep=int(s/iter_number)
            if method=='linear':
                return 1/(iterstep*rate+1)
            if method=='quadratic':
                return 1/(iterstep**0.5*rate+1)
            if method=='exp':
                return torch.exp(-rate*iterstep)
            else:
                return 1
    return f

def gradfilter_ema(
m: nn.Module,
grads: Optional[Dict[str, torch.Tensor]] = None,
alpha: float = 0.98,
lamb: float = 2.0,
 ) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and n in grads:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads
def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 10,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach()) # .cpu())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


def train(config):
    print('using config:', config)
    mode = config['model']['mode']
    optimizer=config['optimizer']
    train_cfg = config['train_'+mode]
    wandb_cfg = config['wandb']
    if wandb_cfg['use_wandb']:
        wandb.init(project=wandb_cfg['wandb_project'], config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    # print(model)
    model.train()
    model.xavier_init()
    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    optimizer_status = train_cfg[optimizer + '_status']
    if 'betas' in optimizer_status:
        optimizer_status['betas'] = tuple(float(beta) for beta in optimizer_status['betas'])
    if optimizer == "GrokAdamW":
        # 使用自定义优化器类
        from optimizer import GrokAdamW
        optimizer_class = GrokAdamW
    else:
        # 使用torch.optim中的类
        optimizer_class = getattr(torch.optim, optimizer)

    optim = optimizer_class(model.parameters(), **optimizer_status)
    SAM=config['using_SAM']
    if SAM=='ASAM':
        optim=ASAM(optim,config['rho'],adaptive=True)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_decay(**train_cfg['lr_decay']))
    step = 0
    grads = None
    for x, y in tqdm(train_dataloader):
        # max_norm = max(param.data.abs().max().item() for param in model.parameters() if param.requires_grad)
        loss, logs = model.get_loss(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        max_norm = max(param.grad.abs().max().item()
                       for group in optim.param_groups
                       for param in group['params']
                       if param.grad is not None and param.grad.abs().max().item() > 0
                       )
        param_norm = math.sqrt(
            sum(
                (param.grad ** 2).sum().item()
                for group in optim.param_groups
                for param in group['params']
                if param.grad is not None and param.grad.abs().max().item() > 0
            )
        )

        # grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
        # grads = gradfilter_ma(model, grads=grads)
        lr_schedule.step()
        if (step+1) % train_cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs = []
                all_test_logs =[ ]
                for i, (val_x, val_y) in tqdm(enumerate(val_dataloader)):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
                for i, (train_x, train_y) in tqdm(enumerate(train_dataloader)):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, test_logs = model.get_loss(train_x.to(device), train_y.to(device))
                    all_test_logs.append(val_logs)
            out_log = {'val': combine_logs(all_val_logs),'test': combine_logs(all_test_logs),'train': combine_logs([logs]), 'step': (step+1),
                       'lr': float(lr_schedule.get_last_lr()[0]), 'l-infinity-norm': max_norm, 'l2-norm':param_norm}
            print(out_log)
            if wandb_cfg['use_wandb']:
                wandb.log(out_log)
            model.train()
        step += 1
        if train_cfg['max_steps'] is not None and step >= train_cfg['max_steps']:
            break


@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
       main()

print(torch.cuda.is_available())  # 如果返回True，则CUDA可用
