import yaml
from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

def log(args, log_path, P=True):
    if P:
        print(args)
    with open(log_path,'a') as f:
        f.write('\n')
        f.write(args)

class LinearLR(_LRScheduler):
    """

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 total_iter,
                 last_epoch=-1):
        self.total_iter = total_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        process = self.last_epoch / self.total_iter
        weight = (1 - process)
        # print('get lr ', [weight * group['initial_lr'] for group in self.optimizer.param_groups])
        return [weight * group['initial_lr'] for group in self.optimizer.param_groups]