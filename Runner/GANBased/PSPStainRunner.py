

import os
import time
import torch
import datetime
import numpy as np 
import torch.distributed as dist
from Register import Registers
from tqdm.autonotebook import tqdm
from src.utils.init_util import init_net
from torch.utils.data import DataLoader
from src.layers.base_layer import get_norm_layer
from src.layers.gauss_pyramid import Gauss_Pyramid_Conv
from Runner.GANBased.GANBaseRunner import GANBaseRunner # type: ignore
from src.models.resnet_normal import ResnetGenerator
from src.models.sampleF_net import PatchSampleF
from src.models.basic_discriminator import NLayerDiscriminator,PixelDiscriminator
from dataset import get_dataset,get_optimizer
from src.losses import GANLoss,PatchNCELoss,MLPA_LOSS,CTPC_LOSS,UNet_pro
from src.utils.args_util import dict2namespace
import src.utils.dist_util as dist_util
from src.utils.img_util import visualize_A2B,tensor_to_image


@Registers.runners.register_with_name("PSPStain_Runner")
class PSPStainRunner(GANBaseRunner):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        
        # modified parameters
        self.lambda_gp = config.lambda_loss_fn.lambda_gp
        self.gp_weight = '[0.015625,0.03125,0.0625,0.125,0.25,1.0]'
        self.asp_loss_mode = 'lambda_linear'

        self.opt = dict2namespace({
            'weight_norm': 'spectral',
            'n_downsampling': self.n_downsampling,
            'nce_T':self.nce_T,
            'batch_size': self.config.data.train.batch_size,
            'asp_loss_mode': self.asp_loss_mode,
            'nce_includes_all_negatives_from_minibatch': self.nce_includes_all_negatives_from_minibatch,
            'n_epochs': self.config.training.n_epochs,
            'n_epochs_decay': self.config.training.n_epochs_decay
            
        })
    
    def train(self):
        pass
    
    