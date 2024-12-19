import torch
from datasets import ModSumDataset, ModSubtractDataset, ModDivisonDataset, PermutationGroup,KSumDataset
from grokk_model import GrokkModel
from utils import convert_path
registry = {}

def register(name):
    def add_f(f):
        registry[name] = f
        return f
    return add_f

def load_item(config, *args, verbose=True):
    config = config.copy()
    name = config.pop('name')
    if name not in registry:
        raise NotImplementedError
    if verbose:
        print(f'loading {name}: {config}')
    return registry[name](config, *args, verbose=verbose)

@register('KSumDataset')
def load_KSumDataset(config, verbose=True):
    return KSumDataset(config['p'], config['num_p'],config['frac_train'],config['seed'])

@register('mod_sum_dataset')
def load_mod_sum_dataset(config, verbose=True):
    return ModSumDataset(config['p'], config['frac_train'],config['seed'])

@register('mod_subtract_dataset')
def load_mod_subtract_dataset(config, verbose=True):
    return ModSubtractDataset(config['p'], config['frac_train'],config['seed'])

@register('mod_division_dataset')
def load_mod_subtract_dataset(config, verbose=True):
    return ModDivisonDataset(config['p'], config['frac_train'])

@register('permutation_group_dataset')
def load_mod_subtract_dataset(config, verbose=True):
    return PermutationGroup(config['k'], config['frac_train'])

@register('grokk_model')
def load_grokk_model(config, vocab_size, out_size, device, verbose=True):
    mode=config['mode']
    model = GrokkModel(config['transformer_config'],config['lstm_config'],config['mlp_config'], vocab_size, out_size, device, mode).to(device)

    if config['checkpoint_path'] is not None:
        print('this test is using',mode)
        if verbose:
            print(f'loading grokk_model state dict from: {convert_path(config["checkpoint_path"])}')
        model.load_state_dict(torch.load(convert_path(config['checkpoint_path']), map_location='cpu'), strict=config['strict_load'])
        if verbose:
            print('loaded.')
    return model

