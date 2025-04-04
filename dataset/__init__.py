
import torch
from Register import Registers
from .folder_dataset import PatchAlignedDataset,BreastPatchAlignedDataset


def get_dataset(data_config, test=False):
    if test:
        test_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, phase='test')
        return None,test_dataset
    train_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, phase='train')
    val_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, phase='test')
    return train_dataset, val_dataset


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))