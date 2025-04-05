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
from src.losses import AdaptiveSupervisedPatchNCELoss,GANLoss,PatchNCELoss
from src.utils.args_util import dict2namespace
import src.utils.dist_util as dist_util
from src.utils.img_util import visualize_A2B,tensor_to_image


@Registers.runners.register_with_name("CUT_Runner")
class CUTRunner(GANBaseRunner):
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
        
        # modified parameters 
        self.n_downsampling = 2
        self.opt = dict2namespace({
            'weight_norm': 'spectral',
            'n_downsampling': self.n_downsampling,
            'nce_T':self.nce_T,
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
                                    ngf=config.model.model_G.ngf,norm_layer=netG_norm,use_dropout=config.model.model_G.no_dropout,n_blocks=9)
            
        elif config.model.model_G.model_name == "resnet_6blocks":
            netG= ResnetGenerator(input_nc=config.model.model_G.input_nc,output_nc=config.model.model_G.output_nc,
                                    ngf=config.model.model_G.ngf,norm_layer=netG_norm,use_dropout=config.model.model_G.no_dropout,n_blocks=6)
            
        init_type = config.model.model_G.init_type
        init_gain = config.model.model_G.init_gan
        self.netG = init_net(netG,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids)
        # default: mlp_smaple ,TODO: other type
        netF = PatchSampleF(use_mlp=True,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids,nc=config.model.model_F.netF_nc)
        self.netF = init_net(netF,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids)

        if isTrain:
            self.model_names.append("D")
            nrom_layer_D = get_norm_layer(norm_type=config.model.model_D.norm)
            netD_name = config.model.model_D.model_name
            input_nc = config.model.model_G.output_nc # netD input_nc = netG_input_nc 
            if netD_name == "basic":
                # PatchGAN
                netD_A = NLayerDiscriminator(input_nc,config.model.model_D.ndf,n_layers=config.model.model_D.n_layer_D,norm_layer=nrom_layer_D)
            elif netD_name == 'pixel':
                netD_A = PixelDiscriminator(input_nc,config.model.model_D.ndf,norm_layer=nrom_layer_D)
            else:
                raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD_name)
            init_type = config.model.model_D.init_type
            init_gain = config.model.model_D.init_gan
            self.netD = init_net(netD_A,init_type,init_gain,gpu_ids=self.gpu_ids)
            return [self.netG,self.netF,self.netD]
        return [self.netG,self.netF]


    def initialize_optimizer_scheduler(self, net,config,is_test=False):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        netG = net[0]
        netF = net[1]
        optimizer_G = get_optimizer(config.model.model_G.optimizer,parameters=netG.parameters())
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_G,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.model_G.lr_scheduler))
        if not is_test:
            netD = net[2]
            optimizer_D = get_optimizer(config.model.model_D.optimizer,parameters=netD.parameters())
            schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_D,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.model_D.lr_scheduler))
            return [optimizer_G,optimizer_D],[schedulerG,schedulerD]
        return [optimizer_G],[schedulerG]

    def train(self):
        print("Running: ",self.__class__.__name__)
        train_dataset,val_dataset = get_dataset(self.config.data,test=False)
        train_loader = DataLoader(train_dataset,
                                    batch_size=self.config.data.train.batch_size,
                                    shuffle=self.config.data.train.shuffle,
                                    num_workers=0,
                                    drop_last=False)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.config.data.val.batch_size,
                                shuffle=self.config.data.val.shuffle,
                                num_workers=0,
                                drop_last=False)
        epoch_length = len(train_loader)
        start_epoch = self.global_epoch
        print(
            f"start training {self.config.model.model_name} on {self.config.data.dataset_name}, {len(train_loader)} iters per epoch")
        # backup config
        self.save_config()

        # create net at gpu_ids[0] GPU
        net,optim,scheduler = self.initialize_model_optimizer_scheduler(self.config,is_test=False)
        self.netG = net[0]
        self.netF = net[1]
        self.netD = net[2]
        self.optim_G = optim[0]
        self.optim_D = optim[1]
        self.lr_schedulerG = scheduler[0]
        self.lr_schedulerD = scheduler[1]
        # loss
        self.criterionGAN = GANLoss('lsgan').to(self.device)
        self.criterionNCE = []
    
    
    def test(self):
        print("Running: ",self.__class__.__name__)
        _,dataset = get_dataset(self.config.data,test=True)
        val_loader = DataLoader(dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    drop_last=False)
        net,_,_ = self.initialize_model_optimizer_scheduler(self.config,is_test=True)
        netG = net[0].to(self.config.training.device[0])
        # load from reslut_path/dataset_name/exp_name/checkpoints/..
        load_path = os.path.join(self.config.result.ckpt_path,f"netG_A2B_latest.pth")
        state_dict = torch.load(load_path, map_location=str(f'{self.config.training.device[0]}'))
        netG.load_state_dict(state_dict)
        netG.eval()
        # create folder for eval and visualize
        realA_paths=os.path.join(self.config.result.image_path,"source")
        realB_paths=os.path.join(self.config.result.image_path,"gt")
        fakeB_paths=os.path.join(self.config.result.image_path,"fake")
        if not os.path.exists(realA_paths):
            os.makedirs(realA_paths)
        if not os.path.exists(realB_paths):
            os.makedirs(realB_paths)
        if not os.path.exists(fakeB_paths):
            os.makedirs(fakeB_paths)
            
        for i,batch in enumerate(val_loader):
            real_A = batch['A'].to(self.config.training.device[0])
            real_B = batch['B'].to(self.config.training.device[0])
            fake_val_B = self.netG(real_A)
            # save A2B
            filename = f"{i:04d}.png" 
            # visualize_A2B(real_A,real_B,fake_val_B,filename,self.config.result.image_path)
            A_img_path = os.path.join(realA_paths,filename)
            B_img_path = os.path.join(realB_paths,filename)
            fake_B_path = os.path.join(fakeB_paths,filename)
            img_A = tensor_to_image(real_A)
            img_B = tensor_to_image(real_B)
            fake_img_B = tensor_to_image(fake_val_B)
            img_A.save(A_img_path)
            img_B.save(B_img_path)
            fake_img_B.save(fake_B_path)
            print(f"save image {i:04d}.png")