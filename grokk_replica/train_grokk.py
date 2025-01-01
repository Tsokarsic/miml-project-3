import torch
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import IterableDataset

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
    if config['init_method'] != 'default':
        init_method=getattr(model,config['init_method']+'_init')
        init_method()
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
        l2_norm = torch.sqrt(sum(torch.sum(param.grad ** 2)
                                                 for group in optim.param_groups
                                                 for param in group['params']
                                                 if param.grad is not None and param.grad.abs().max().item() > 0))
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
                       'lr': float(lr_schedule.get_last_lr()[0]), 'l-infinity-norms': max_norm,'l2-norm':l2_norm}
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

# def calculate_edge_popup_scores(model):
#     """
#     Calculate Edge Popup scores for all trainable parameters: |W| * grad(W)
#
#     Args:
#         model: The PyTorch model.
#
#     Returns:
#         scores: Dictionary of scores per layer.
#     """
#     scores = {}
#     for name, param in model.named_parameters():
#         if 'weight' in name and param.requires_grad and param.grad is not None:
#             # Calculate |W| * |grad(W)| (element-wise product)
#             scores[name] = (param.data.abs() * param.grad.abs()).detach()
#     return scores
#
# def apply_topk_mask(scores, sparsity):
#     """
#     Apply Top-k algorithm to prune weights based on sparsity.
#
#     Args:
#         scores: Dict of importance scores for each layer (from calculate_edge_popup_scores).
#         sparsity: Target sparsity level (0 < sparsity < 1).
#
#     Returns:
#         mask: Dictionary of layer-wise binary masks.
#     """
#     mask = {}
#     for name, score in scores.items():
#         flattened_scores = score.view(-1)  # Flatten to 1D
#         k = int(flattened_scores.numel() * (1 - sparsity))  # Number of weights to retain
#         threshold, _ = torch.kthvalue(flattened_scores, k)  # Find the cut-off threshold
#
#         # Generate binary mask: 1 if score >= threshold, else 0
#         mask[name] = (score >= threshold).float().view(score.size())
#     return mask
#
# def train_with_edge_popup(model, dataloader, device, optimizer, sparsity, total_steps):
#     """
#     Train the model using Edge Popup algorithm.
#
#     Args:
#         model: PyTorch model to train.
#         dataloader: DataLoader for training data.
#         device: Torch device (CPU/GPU).
#         optimizer: Optimizer.
#         sparsity: Sparsity level (e.g., 0.5 = 50% weights pruned).
#         total_steps: Total number of steps for training.
#     """
#     mask = None  # Initialize Edge Popup mask
#     step = 0
#
#     for x, y in tqdm(dataloader):
#         model.train()
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#
#         # Forward pass
#         loss, _ = model.get_loss(x, y)
#         loss.backward()
#
#         # Step 1: Calculate scores and update mask every step
#         scores = calculate_edge_popup_scores(model)
#         mask = apply_topk_mask(scores, sparsity)
#
#         # Step 2: Apply mask to enforce sparsity (weights * mask)
#         for name, param in model.named_parameters():
#             if name in mask:
#                 param.data.mul_(mask[name])  # Enforce the mask
#
#         # Step 3: Update optimizer
#         optimizer.step()
#
#         # Step counter
#         step += 1
#         if step >= total_steps:
#             break

# def train(config):
#     print("using config:", config)
#     mode = config["model"]["mode"]
#     optimizer = config["optimizer"]
#     train_cfg = config["train_" + mode]
#     wandb_cfg = config["wandb"]
#
#     # Edge Popup 配置
#     use_edge_popup = config.get("use_edge_popup", False)  # 是否启用 Edge Popup
#     sparsity = config.get("sparsity", 0.5)  # Edge Popup 稀疏比例
#     max_edge_popup_steps = config.get("max_edge_popup_steps", 1000)  # 稀疏化最大步骤
#     accuracy_threshold = config.get("accuracy_threshold", 1.0)  # 切换稀疏化的准确率阈值
#     post_threshold_wait_steps = config.get("post_threshold_wait_steps", 10 * train_cfg["eval_every"])  # 阈值后等待
#
#     # 初始化 WandB
#     if wandb_cfg["use_wandb"]:
#         wandb.init(project=wandb_cfg["wandb_project"], config=config)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 数据加载与模型初始化
#     dataset = load_item(config["dataset"])
#     train_data = GroupDataset(dataset, "train")
#     val_data = GroupDataset(dataset, "val")
#     model = load_item(config["model"], dataset.n_vocab, dataset.n_out, device)
#     model.train()
#     if config['init_method'] != 'default':
#         init_method=getattr(model,config['init_method']+'_init')
#         init_method()
#
#     train_dataloader = DataLoader(train_data, num_workers=train_cfg["num_workers"], batch_size=train_cfg["bsize"])
#     val_dataloader = DataLoader(val_data, num_workers=train_cfg["num_workers"], batch_size=train_cfg["bsize"])
#
#     # 初始化优化器
#     optimizer_status = train_cfg[optimizer + "_status"]
#     if "betas" in optimizer_status:
#         optimizer_status["betas"] = tuple(float(beta) for beta in optimizer_status["betas"])
#     # if optimizer == "GrokAdamW":
#     #     from optimizer import GrokAdamW
#     #
#     #     optimizer_class = GrokAdamW
#     else:
#         optimizer_class = getattr(torch.optim, optimizer)
#
#     optim = optimizer_class(model.parameters(), **optimizer_status)
#
#     SAM = config["using_SAM"]
#     if SAM == "ASAM":
#         optim = ASAM(optim, config["rho"], adaptive=True)
#     lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_decay(**train_cfg["lr_decay"]))
#
#     # 初始化训练状态
#     step = 0
#     achieved_train_accuracy = False
#     post_threshold_countdown = post_threshold_wait_steps
#     edge_popup_steps = 0
#     edge_popup_active = False
#     mask = {}  # Edge Popup 稀疏化掩码
#
#     # 训练主循环
#     while step < train_cfg["max_steps"]:
#         for x, y in train_dataloader:
#             x, y = x.to(device), y.to(device)
#             # grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
#             # grads = gradfilter_ma(model, grads=grads)
#             if not edge_popup_active:
#                 # ===== 普通训练阶段 =====
#                 loss, logs = model.get_loss(x, y)
#                 optim.zero_grad()
#                 loss.backward()
#                 optim.step()
#                 lr_schedule.step()
#                 max_norm = max(param.grad.abs().max().item()
#                                for group in optim.param_groups
#                                for param in group['params']
#                                if param.grad is not None and param.grad.abs().max().item() > 0
#                                )
#                 l2_norm = torch.sqrt(sum(torch.sum(param.grad ** 2)
#                                          for group in optim.param_groups
#                                          for param in group['params']
#                                          if param.grad is not None and param.grad.abs().max().item() > 0))
#             else:
#                 # ===== Edge Popup 阶段 =====
#                 scores = calculate_edge_popup_scores(model)  # 计算剪枝分数
#                 mask = apply_topk_mask(scores, sparsity)  # 稀疏化逻辑更新掩码
#
#                 with torch.no_grad():
#                     for name, param in model.named_parameters():
#                         if name in mask:
#                             param.mul_(mask[name])  # 应用稀疏化掩码，仅更新重要权重
#                             param.grad = None
#                 max_norm = max(param.grad.abs().max().item()
#                                for group in optim.param_groups
#                                for param in group['params']
#                                if param.grad is not None and param.grad.abs().max().item() > 0
#                                )
#                 # 禁止对非稀疏权重计算梯度
#
#             # ===== 周期性评估 =====
#             if (step + 1) % train_cfg["eval_every"] == 0:
#                 train_acc = model.get_train_accuracy(x,y)
#                 print(f"[Step {step + 1}] Train Accuracy: {train_acc:.4f}")
#
#                 if use_edge_popup and not achieved_train_accuracy and train_acc >= accuracy_threshold:
#                     print(f"Train Accuracy reached {train_acc:.4f}, starting countdown.")
#                     achieved_train_accuracy = True
#
#                 if use_edge_popup and achieved_train_accuracy and post_threshold_countdown > 0:
#                     post_threshold_countdown -= train_cfg["eval_every"]
#                     print(f"Waiting... {post_threshold_countdown} steps before entering Edge Popup.")
#
#                 # 当倒计时完成时，启用 Edge Popup
#                 if achieved_train_accuracy and post_threshold_countdown <= 0:
#                     print("Switching to Edge Popup phase!")
#                     edge_popup_active = True
#
#             # ===== 验证阶段 =====
#             if (step + 1) % train_cfg["eval_every"] == 0:
#                 model.eval()
#                 with torch.no_grad():
#                     all_val_logs = []
#                     all_test_logs =[ ]
#                     for i, (val_x, val_y) in tqdm(enumerate(val_dataloader)):
#                         if i >= train_cfg['eval_batches']:
#                             break
#                         _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
#                         all_val_logs.append(val_logs)
#                     for i, (train_x, train_y) in tqdm(enumerate(train_dataloader)):
#                         if i >= train_cfg['eval_batches']:
#                             break
#                         _, test_logs = model.get_loss(train_x.to(device), train_y.to(device))
#                         all_test_logs.append(val_logs)
#                 out_log = {'val': combine_logs(all_val_logs),'test': combine_logs(all_test_logs),'train': combine_logs([logs]), 'step': (step+1),
#                            'lr': float(lr_schedule.get_last_lr()[0]), 'l-infinity-norms': max_norm,'l2-norms': l2_norm}
#                 print(out_log)
#                 if wandb_cfg["use_wandb"]:
#                     wandb.log(out_log)
#                 model.train()
#
#             # 更新训练步数
#             step += 1
#             if step >= train_cfg["max_steps"]:
#                 break
#
#         # Edge Popup 训练完成退出条件
#         if edge_popup_active and edge_popup_steps >= max_edge_popup_steps:
#             print("Edge Popup phase completed. Stopping training.")
#             return
#
# def gradfilter_ema(
# m: nn.Module,
# grads: Optional[Dict[str, torch.Tensor]] = None,
# alpha: float = 0.98,
# lamb: float = 2.0,
#  ) -> Dict[str, torch.Tensor]:
#     if grads is None:
#         grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
#
#     for n, p in m.named_parameters():
#         if p.requires_grad and n in grads:
#             grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
#             p.grad.data = p.grad.data + grads[n] * lamb
#
#     return grads
# def gradfilter_ma(
#     m: nn.Module,
#     grads: Optional[Dict[str, deque]] = None,
#     window_size: int = 10,
#     lamb: float = 5.0,
#     filter_type: Literal['mean', 'sum'] = 'mean',
#     warmup: bool = True,
#     trigger: bool = False, # For ablation study.
# ) -> Dict[str, deque]:
#     if grads is None:
#         grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
#
#     for n, p in m.named_parameters():
#         if p.requires_grad and p.grad is not None:
#             grads[n].append(p.grad.data.detach()) # .cpu())
#
#             # Modify the gradients.
#             if not warmup or len(grads[n]) == window_size and not trigger:
#                 if filter_type == "mean":
#                     avg = sum(grads[n]) / len(grads[n])
#                 elif filter_type == "sum":
#                     avg = sum(grads[n])
#                 else:
#                     raise ValueError(f"Unrecognized filter_type {filter_type}")
#                 p.grad.data = p.grad.data + avg * lamb
#
#     return grads
