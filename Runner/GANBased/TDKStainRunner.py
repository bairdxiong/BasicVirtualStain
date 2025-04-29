

import os
import time
import torch
import datetime
import numpy as np 
import torch.distributed as dist
from Register import Registers
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from src.utils.init_util import init_net
from torch.utils.data import DataLoader
from src.layers.base_layer import get_norm_layer
from src.layers.gauss_pyramid import Gauss_Pyramid_Conv
from Runner.GANBased.GANBaseRunner import GANBaseRunner # type: ignore
from src.models.resnet_normal import ResnetGenerator
from src.models.sampleF_net import PatchSampleF
from src.models.nucleidensity_estimator import NucleiDensityMapEstimator
from src.models.basic_discriminator import NLayerDiscriminator,PixelDiscriminator,MultiscaleDiscriminator
from dataset import get_dataset,get_optimizer
from src.losses import PatchNCELoss
from src.losses.gan_loss import GANLoss4MultiSclae as GANLoss
from src.utils.args_util import dict2namespace
import src.utils.dist_util as dist_util
from src.utils.img_util import visualize_A2B,tensor_to_image


@Registers.runners.register_with_name("TDKStain_Runner")
class TDKStainRunner(GANBaseRunner):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        
        self.lambda_GAN= config.lambda_loss_fn.lambda_GAN
        self.lambda_L1= config.lambda_loss_fn.lambda_L1
        self.lambda_mask= config.lambda_loss_fn.lambda_mask
        self.lambda_E= config.lambda_loss_fn.lambda_E
        self.lambda_nuclei= config.lambda_loss_fn.lambda_nuclei
        # modified parameters
        self.n_downsampling = 3
        self.feat_layers = []
        for i in range(1, self.n_downsampling + 1):
            self.feat_layers.append(4 + i * 3 - 1)
            
        self.opt = dict2namespace({
            'weight_norm': 'spectral',
            'n_downsampling': self.n_downsampling,
            'n_epochs': self.config.training.n_epochs,
            'n_epochs_decay': self.config.training.n_epochs_decay
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
        
        
        feats_nc = 0
        for nc in range(1, self.opt.n_downsampling + 1):
            feats_nc += self.config.model.model_G.ngf * (2 ** nc)
        netE_norm = get_norm_layer(norm_type=config.model.model_E.norm)
        netE = NucleiDensityMapEstimator(input_nc=feats_nc,output_nc=1,
                                         nef=config.model.model_E.nef,n_blocks=config.model.model_E.n_blocks,norm_layer=netE_norm,
                                         use_dropout=config.model.model_E.use_dropout,opt=self.opt)
        init_type = config.model.model_E.init_type
        init_gain = config.model.model_E.init_gan
        self.netE = init_net(netE,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids)
        
        if isTrain:
            self.model_names.append("D")
            nrom_layer_D = get_norm_layer(norm_type=config.model.model_D.norm)
            netD_name = config.model.model_D.model_name
            input_nc = config.model.model_G.output_nc+config.model.model_G.input_nc # netD input_nc = netG_input_nc 
            if netD_name in ["basic",'pixel']: 
                # PatchGAN
                netD_A = MultiscaleDiscriminator(input_nc,config.model.model_D.ndf,
                                                 netD=netD_name,n_layers_D=config.model.model_D.n_layer_D,
                                                 norm_layer=nrom_layer_D,num_D=config.model.model_D.num_D,opt=self.opt)
            else:
                raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD_name)
            init_type = config.model.model_D.init_type
            init_gain = config.model.model_D.init_gan
            self.netD = init_net(netD_A,init_type,init_gain,gpu_ids=self.gpu_ids)
            return [self.netG,self.netE,self.netD]
        return [self.netG,self.netE]

    

    def initialize_optimizer_scheduler(self, net,config,is_test=False):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + (self.global_epoch+1) - config.training.n_epochs) / float(config.training.n_epochs_decay + 1)
            return lr_l
        
        netG = net[0]
        netE = net[1]
        optimizer_G = get_optimizer(config.model.model_G.optimizer,parameters=netG.parameters())
        if config.model.model_G.lr_scheduler.type=='linear':
            schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_G,lr_lambda=lambda_rule)
        else:
            from types import SimpleNamespace
            del config.model.model_G.lr_scheduler.type
            schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_G,
                                                                mode='min',
                                                                verbose=True,
                                                                threshold_mode='rel',
                                                                **vars(config.model.model_G.lr_scheduler))
        if not is_test:
            netD = net[2]
            optimizer_D = get_optimizer(config.model.model_D.optimizer,parameters=netD.parameters())
            if config.model.model_D.lr_scheduler.type=='linear':
                schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_D,lr_lambda=lambda_rule)
            else:
                from types import SimpleNamespace
                del config.model.model_D.lr_scheduler.type
                schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_D,
                                                                   mode='min',
                                                                     verbose=True,
                                                                   threshold_mode='rel',
                                                                   **vars(config.model.model_D.lr_scheduler))
            optimizer_E =get_optimizer(config.model.model_E.optimizer,parameters=netE.parameters())
            if config.model.model_E.lr_scheduler.type=='linear':
                schedulerE = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_E,lr_lambda=lambda_rule)
            else:
                from types import SimpleNamespace
                del config.model.model_D.lr_scheduler.type
                schedulerE = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_E,
                                                                   mode='min',
                                                                     verbose=True,
                                                                   threshold_mode='rel',
                                                                   **vars(config.model.model_E.lr_scheduler))
            return [optimizer_G,optimizer_D,optimizer_E],[schedulerG,schedulerD,schedulerE]
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
        self.isTrain = True
        # create net at gpu_ids[0] GPU
        net,optim,scheduler = self.initialize_model_optimizer_scheduler(self.config,is_test=False)
        self.netG = net[0]
        self.netE = net[1]
        self.netD = net[2]
        self.optim_G = optim[0]
        self.optim_D = optim[1]
        self.optim_E = optim[2]
        self.lr_schedulerG = scheduler[0]
        self.lr_schedulerD = scheduler[1]
        self.lr_schedulerE = scheduler[2]
        # loss
        self.criterionGAN = GANLoss(self.config.lambda_loss_fn.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMask = torch.nn.L1Loss()
        self.criterionEstimator = torch.nn.MSELoss()
        self.criterionNuclei = torch.nn.MSELoss()
        
        # Resume training if a checkpoint path is provided
        if not self.config.training.resume_checkpoint ==  "None":
            self.resume_training(self.config.training.resume_checkpoint)
            
        for epoch in range(self.global_epoch,self.config.training.n_epochs):
            if self.global_epoch > self.config.training.n_epochs:
                break
            pbar = tqdm(train_loader,total=len(train_loader),smoothing=0.01)
            self.global_epoch = epoch
            start_time = time.time()
            self.current_epoch = epoch
            for train_batch in pbar:
                self.global_step +=1
                self.real_A = train_batch['A'].to(self.config.training.device[0]) 
                self.real_B = train_batch['B'].to(self.config.training.device[0])
                self.real_B_dab = train_batch['DAB'].to(self.config.training.device[0])
                self.dab_mask = train_batch['DAB_Mask'].to(self.config.training.device[0])
                self.true_nuclei = train_batch['Nuclei_map'].to(self.config.training.device[0])
                self.forward()
                # uppdate E 
                self.set_requires_grad(self.netG,False)
                self.set_requires_grad(self.netE,True)
                self.optim_E.zero_grad()
                self.backward_E()
                self.optim_E.step()
                # update D 
                self.set_requires_grad(self.netD,True)
                self.optim_D.zero_grad()
                self.backward_D()
                self.optim_D.step()
                # update G
                self.set_requires_grad(self.netG,True)
                self.set_requires_grad(self.netD,False)
                self.set_requires_grad(self.netE,False)
                self.optim_G.zero_grad()
                self.backward_G()
                self.optim_G.step()

                pbar.set_description((
                    f"Epoch:[{epoch+1} / {self.config.training.n_epochs}] "
                    f" iter:{self.global_step} loss_D:{self.loss_D.detach().mean()},loss_G:{self.loss_G.detach().mean()}"))
                pbar.update(1)

                # save image grid each x iterations
                with torch.no_grad():
                    if self.global_step % self.config.training.save_interval ==0: #
                        val_batch = next(iter(val_loader))
                        val_A = val_batch['A']
                        val_B = val_batch['B']
                        val_A=val_A.to(self.config.training.device[0])
                        val_B=val_B.to(self.config.training.device[0])
                        self.netG.eval()
                        fake_val_B = self.netG(val_A)
                        self.netG.train()
                        # save A2B
                        filename = f"val_A2B_{self.global_step}.png" 
                        visualize_A2B(val_A,val_B,fake_val_B,filename,self.config.result.log_path)

            pbar.close()
            end_time = time.time()
            elapsed_rounded = int(round((end_time-start_time)))
            print(f"Epoch: [{epoch+1} / {self.config.training.n_epochs}] training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))
            
            # save checkpoint
            if (epoch+1)%self.config.training.save_pth_interval==0 or \
                (epoch + 1) == self.config.training.n_epochs:
                with torch.no_grad():
                    print("saving latest checkpoint....")
                    self.save_checkpoint(epoch+1,os.path.join(self.config.result.ckpt_path,f"netG_A2B_{epoch+1}.pth"))
                    # torch.save(self.netG.state_dict(),os.path.join(self.config.result.ckpt_path,f"netG_A2B_{epoch+1}.pth"))
            # torch.save(self.netG.state_dict(),os.path.join(self.config.result.ckpt_path,f"netG_A2B_latest.pth"))
            self.save_checkpoint(epoch+1,os.path.join(self.config.result.ckpt_path,f"netG_A2B_latest.pth"))


    
    def forward(self):
        self.fake_B = self.netG(self.real_A)
        self.fake_B_dab = self.netG(self.real_A*self.dab_mask)
    
    def backward_D(self):
        fake_AB = torch.cat((self.real_A,self.fake_B),1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake,False)
        real_AB = torch.cat((self.real_A,self.real_B),1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real,True)
        self.loss_D = (self.loss_D_real+self.loss_D_fake)*0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_AB = torch.cat((self.real_A,self.fake_B),1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake,True)
        self.loss_G_L1 = self.criterionL1(self.fake_B,self.real_B)*self.lambda_L1
        self.loss_G_mask = self.criterionMask(self.fake_B_dab,self.real_B_dab)*self.lambda_mask
        self.calculate_nuclei_density_loss()
        # combine loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_mask + self.loss_nuclei
        # calculate gradients
        self.loss_G.backward()
        
    def calculate_nuclei_density_loss(self):
        self.fake_B_feats = self.netG(self.fake_B, layers=self.feat_layers, encode_only=True)
        self.fake_B_nuclei = self.netE(self.fake_B_feats)
        self.fake_B_nuclei = F.interpolate(self.fake_B_nuclei, size=self.true_nuclei.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_nuclei = self.criterionNuclei(self.fake_B_nuclei, self.true_nuclei) * self.lambda_nuclei
    
    def backward_E(self):
        self.real_B_feats = self.netG(self.real_B, layers=self.feat_layers, encode_only=True)
        self.real_B_nuclei = self.netE(self.real_B_feats)
        self.real_B_nuclei = F.interpolate(self.real_B_nuclei, size=self.true_nuclei.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_E = self.criterionEstimator(self.real_B_nuclei, self.true_nuclei) * self.lambda_E
        self.loss_E.backward()
    
    # rewirte save checkpoint
    def save_checkpoint(self,epoch, path):
        """
        保存训练状态，包括模型权重、优化器状态、调度器状态和训练进度。
        """
        checkpoint = {
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'netE_state_dict': self.netE.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'optim_E_state_dict': self.optim_E.state_dict(),
            'scheduler_G_state_dict': self.lr_schedulerG.state_dict(),
            'scheduler_D_state_dict': self.lr_schedulerD.state_dict(),
            'scheduler_E_state_dict': self.lr_schedulerE.state_dict(),
            'epoch': epoch,
            'step': self.global_step,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")
    
    
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
        load_path = os.path.join(self.config.result.ckpt_path,f"netG_A2B_latest.pth") #
        state_dict = torch.load(load_path, map_location=str(f'{self.config.training.device[0]}'))['netG_state_dict']
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