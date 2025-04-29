"""
heavily inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py
"""
import os
import yaml
import torch
import argparse
import shutil
from abc import ABC
from PIL import Image
from tqdm.autonotebook import tqdm
from abc import ABC, abstractmethod
# from Runners.utils import make_save_dirs, make_dir, get_dataset, remove_file
from dataset import get_dataset
from src.utils.save_util import make_save_dirs

class GANBaseRunner(ABC):
    def __init__(self,config):

        self.net = None 
        self.model_names = []
        self.loss_names = []
        self.config = config
        self.optimizer = None
        self.gpu_ids = list(map(int,config.args.gpu_ids.split(',')))

        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.image_path, \
        self.config.result.ckpt_path, \
        self.config.result.log_path, \
        self.config.result.sample_path, \
        self.config.result.sample_to_eval_path = make_save_dirs(self.config.args,
                                                                prefix=self.config.data.dataset_name,
                                                                suffix=self.config.model.model_name)
        
        self.global_epoch = 0  # global epoch
        self.global_step = 0
    
    def resume_training(self, checkpoint_path):
        """
        恢复训练状态，包括模型权重、优化器状态、调度器状态和训练进度。
        """
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
        self.optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
        self.lr_schedulerG.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.lr_schedulerD.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.global_epoch = checkpoint['epoch']
        self.global_step = checkpoint['step']
        print(f"Checkpoint loaded: epoch={self.global_epoch}, step={self.global_step}")

    def save_checkpoint(self, epoch, path):
        """
        保存训练状态，包括模型权重、优化器状态、调度器状态和训练进度。
        """
        checkpoint = {
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'scheduler_G_state_dict': self.lr_schedulerG.state_dict(),
            'scheduler_D_state_dict': self.lr_schedulerD.state_dict(),
            'epoch': epoch,
            'step': self.global_step,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")
    
    @abstractmethod
    def initialize_model(self, config,is_test):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass
    
    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config,is_test):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config,is_test)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config,is_test=is_test)
        return net, optimizer, scheduler
    
    # save configuration file
    def save_config(self):
        # eval/test neednt backup yaml
        if self.config.args.sample_to_eval:
            return
            
        backup_path = os.path.join(self.config.result.ckpt_path, 'config_backup.yaml')
        shutil.copyfile(self.config.args.config, backup_path)

        save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)
    
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
    
    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad