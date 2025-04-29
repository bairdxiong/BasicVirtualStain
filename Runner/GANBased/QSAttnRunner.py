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
from src.models.sampleF_net import PatchAttnSampleF
from src.models.basic_discriminator import NLayerDiscriminator,PixelDiscriminator
from dataset import get_dataset,get_optimizer
from src.losses import GANLoss,PatchNCELoss
from src.utils.args_util import dict2namespace
import src.utils.dist_util as dist_util
from src.utils.img_util import visualize_A2B,tensor_to_image
from Runner.GANBased.CUTRunner import CUTRunner


@Registers.runners.register_with_name("QSAttn_Runner")
class QSAttnRunner(CUTRunner):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.lambda_GAN = 1.0
        self.lambda_NCE = 10.0
        self.nce_T = config.model.nce_T
        self.num_patches=config.model.num_patches
        nce_layers = config.model.nce_layers
        self.nce_layers=[int(i) for i in nce_layers.split(',')] # compute NCE loss on which layers
        self.flip_equivariance = config.model.flip_equivariance
        self.nce_idt = config.model.nce_idt  # use NCE loss for identity mapping NCE(G(Y),Y)
        self.nce_includes_all_negatives_from_minibatch=config.model.nce_includes_all_negatives_from_minibatch
        self.device = self.config.training.device[0]
        
        # scheduler
        self.n_epochs_decay = config.training.n_epochs_decay
        
        # modified parameters 
        self.n_downsampling = 2
        self.opt = dict2namespace({
            'weight_norm': 'spectral',
            'n_downsampling': self.n_downsampling,
            'nce_T':self.nce_T,
            'batch_size': self.config.data.train.batch_size,
            'nce_includes_all_negatives_from_minibatch': self.nce_includes_all_negatives_from_minibatch,
        })
        
        # use ddp or not
        if config.training.use_DDP:
            rank = dist.get_rank()
            self.device = rank%torch.cuda.device_count()
        else:
            self.device = self.config.training.device[0]#dist_util.dev()
        
    
    def initialize_model(self, config,is_test=False):
        """
        netG & netF, netD(G) & netD(F)
        initialize model
        :param config: config
        :return: nn.Module
        """
        isTrain= not is_test
        
        netG_norm = get_norm_layer(norm_type=config.model.model_G.norm)
        if config.model.model_G.model_name == "resnet_9blocks":
            netG= ResnetGenerator(input_nc=config.model.model_G.input_nc,output_nc=config.model.model_G.output_nc,
                                    ngf=config.model.model_G.ngf,norm_layer=netG_norm,use_dropout=config.model.model_G.no_dropout,n_blocks=9,opt=self.opt)
            
        elif config.model.model_G.model_name == "resnet_6blocks":
            netG= ResnetGenerator(input_nc=config.model.model_G.input_nc,output_nc=config.model.model_G.output_nc,
                                    ngf=config.model.model_G.ngf,norm_layer=netG_norm,use_dropout=config.model.model_G.no_dropout,n_blocks=6,opt=self.opt)
            
        init_type = config.model.model_G.init_type
        init_gain = config.model.model_G.init_gan
        self.netG = init_net(netG,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids)
        # default: mlp_smaple ,TODO: other type
        netF = PatchAttnSampleF(use_mlp=True,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids,nc=config.model.model_F.netF_nc)
        self.netF = init_net(netF,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids)

        if isTrain:
            self.model_names.append("D")
            nrom_layer_D = get_norm_layer(norm_type=config.model.model_D.norm)
            netD_name = config.model.model_D.model_name
            input_nc = config.model.model_G.output_nc # netD input_nc = netG_input_nc 
            if netD_name == "basic":
                # PatchGAN
                netD_A = NLayerDiscriminator(input_nc,config.model.model_D.ndf,n_layers=config.model.model_D.n_layer_D,norm_layer=nrom_layer_D,opt=self.opt)
            elif netD_name == 'pixel':
                netD_A = PixelDiscriminator(input_nc,config.model.model_D.ndf,norm_layer=nrom_layer_D,opt=self.opt)
            else:
                raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD_name)
            init_type = config.model.model_D.init_type
            init_gain = config.model.model_D.init_gan
            self.netD = init_net(netD_A,init_type,init_gain,gpu_ids=self.gpu_ids)
            return [self.netG,self.netF,self.netD]
        return [self.netG,self.netF]

    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, nce_layers):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.flip_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids,attn_mats = netF(feat_k, self.num_patches, None,None)
        feat_q_pool, _,_ = netF(feat_q, self.num_patches, sample_ids,attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers