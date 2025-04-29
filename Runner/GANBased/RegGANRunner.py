
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
from src.models.reg_model import Reg,Transformer_2D
from src.losses.smooth_loss import Grad

@Registers.runners.register_with_name("RegGAN_Runner")
class RegGANRunner(GANBaseRunner):
    def __init__(self,config):
        super().__init__(config)
        self.config = config

        self.lambda_Adv = self.config.lambda_loss_fn.lambda_GAN
        self.lambda_Cyc = self.config.lambda_loss_fn.lambda_Cyc
        self.lambda_Corr = self.config.lambda_loss_fn.lambda_Corr
        self.lambda_smooth = self.config.lambda_loss_fn.lambda_Smooth
        self.lambda_NCE = 10.0
        self.nce_T = config.model.nce_T
        self.num_patches=config.model.num_patches
        nce_layers = config.model.nce_layers
        self.nce_layers=[int(i) for i in nce_layers.split(',')] # compute NCE loss on which layers
        self.flip_equivariance = config.model.flip_equivariance
        self.nce_idt = config.model.nce_idt  # use NCE loss for identity mapping NCE(G(Y),Y)
        self.nce_includes_all_negatives_from_minibatch=config.model.nce_includes_all_negatives_from_minibatch
        self.n_downsampling = 2
        self.opt={
            'weight_norm': 'spectral',
            'n_downsampling': self.n_downsampling,
            'nce_T':self.nce_T,
            'batch_size': self.config.data.train.batch_size,
            'nce_includes_all_negatives_from_minibatch': self.nce_includes_all_negatives_from_minibatch,
            
        }
        self.opt = dict2namespace(self.opt)
        # use ddp or not
        if config.training.use_DDP:
            rank = dist.get_rank()
            self.device = rank%torch.cuda.device_count()
        else:
            self.device = self.config.training.device[0]#dist_util.dev()
    
    def initialize_model(self, config,is_test):
        isTrain = not is_test
        netG_norm = get_norm_layer(norm_type=config.model.model_G.norm)
        if config.model.model_G.model_name == "resnet_9blocks":
            # RegGAN use no_antialias=True no_antialias_up=True
            netG= ResnetGenerator(input_nc=config.model.model_G.input_nc,output_nc=config.model.model_G.output_nc,
                                    ngf=config.model.model_G.ngf,norm_layer=netG_norm,use_dropout=config.model.model_G.no_dropout,n_blocks=9,opt=self.opt,)
            
        elif config.model.model_G.model_name == "resnet_6blocks":
            netG= ResnetGenerator(input_nc=config.model.model_G.input_nc,output_nc=config.model.model_G.output_nc,
                                    ngf=config.model.model_G.ngf,norm_layer=netG_norm,use_dropout=config.model.model_G.no_dropout,n_blocks=6,opt=self.opt)
        init_type = config.model.model_G.init_type
        init_gain = config.model.model_G.init_gan
        self.netG = init_net(netG,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids)
        
        netF = PatchSampleF(use_mlp=True,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids,nc=config.model.model_F.netF_nc)
        self.netF = init_net(netF,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids)
        
        if isTrain:
            self.model_names.append("D")
            nrom_layer_D = get_norm_layer(norm_type=config.model.model_D.norm)
            netD_name = config.model.model_D.model_name
            input_nc = config.model.model_G.output_nc+config.model.model_G.input_nc # netD input_nc = netG_input_nc 
            if netD_name == "basic":
                # PatchGAN
                netD_A = NLayerDiscriminator(input_nc,config.model.model_D.ndf,n_layers=config.model.model_D.n_layer_D,norm_layer=nrom_layer_D,no_antialias=False,opt=self.opt)
            elif netD_name == 'pixel':
                netD_A = PixelDiscriminator(input_nc,config.model.model_D.ndf,norm_layer=nrom_layer_D)
            else:
                raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD_name)
            init_type = config.model.model_D.init_type
            init_gain = config.model.model_D.init_gan
            self.netD = init_net(netD_A,init_type,init_gain,gpu_ids=self.gpu_ids)
            
            # Reg
            self.netR = Reg(height=self.config.data.dataset_config.load_size,width=self.config.data.dataset_config.load_size,in_channels_a=self.config.model.model_G.output_nc,in_channels_b=self.config.model.model_G.input_nc)
            self.netR = init_net(self.netR,init_type,init_gain,gpu_ids=self.gpu_ids)
            self.stn = Transformer_2D(self.device)
            
            
            return [self.netG,self.netD,self.netR,self.stn,self.netF]
        return [self.netG,self.netF]
    
    def initialize_optimizer_scheduler(self, net, config,is_test):
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
        optimizer_G = get_optimizer(config.model.model_G.optimizer,parameters=netG.parameters())
        if config.model.model_G.lr_scheduler.type=='linear':
            schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_G,lr_lambda=lambda_rule)
        else:
            schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_G,
                                                                mode='min',
                                                                verbose=True,
                                                                threshold_mode='rel',
                                                                **vars(config.model.model_G.lr_scheduler))
        if not is_test:
            netD = net[1]
            optimizer_D = get_optimizer(config.model.model_D.optimizer,parameters=netD.parameters())
            if config.model.model_D.lr_scheduler.type=='linear':
                schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_D,lr_lambda=lambda_rule)
            else:
                schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_D,
                                                                   mode='min',
                                                                   verbose=True,
                                                                   threshold_mode='rel',
                                                                   **vars(config.model.model_D.lr_scheduler))
            netR = net[2]
            optimizer_R = get_optimizer(config.model.model_R.optimizer,parameters=netR.parameters())
            if config.model.model_R.lr_scheduler.type=='linear':
                schedulerR = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_R,lr_lambda=lambda_rule)
            else:
                schedulerR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_R,
                                                                   mode='min',
                                                                   verbose=True,
                                                                   threshold_mode='rel',
                                                                   **vars(config.model.model_D.lr_scheduler))
            return [optimizer_G,optimizer_D,optimizer_R],[schedulerG,schedulerD,schedulerR]
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
        self.netD = net[1]
        self.netR = net[2]
        self.stn = net[3]
        self.netF = net[4]
        self.optim_G = optim[0]
        self.optim_D = optim[1]
        self.optim_R = optim[2]
        self.lr_schedulerG = scheduler[0]
        self.lr_schedulerD = scheduler[1]
        self.lr_schedulerR = scheduler[2]
        # loss
        self.criterionGAN = GANLoss(self.config.lambda_loss_fn.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionNCE = []
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(self.opt).to(self.device))
        
        first_step = False
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
                if epoch == start_epoch and not first_step:
                    bs_per_gpu = self.real_A.size(0)  # assume single gpu 
                    self.real_A = self.real_A[:bs_per_gpu]
                    self.real_B = self.real_B[:bs_per_gpu]
                    self.forward()
                    self.compute_D_loss().backward()                  # calculate gradients for D
                    loss_R = self.compute_R_loss()
                    loss_G = self.compute_G_loss()               # calculate graidents for G
                    loss_G = loss_G + loss_R 
                    loss_G.backward()
                    if self.lambda_NCE>0.0:
                        optimizer_F = get_optimizer(self.config.model.model_G.optimizer,self.netF.parameters())
                    first_step = True
                self.forward()
                self.set_requires_grad(self.netD,True)
                self.optim_D.zero_grad()
                self.loss_D = self.compute_D_loss()
                self.loss_D.backward()
                self.optim_D.step()
                # update G
                self.set_requires_grad(self.netD,False)
                self.optim_G.zero_grad()
                self.optim_R.zero_grad()
                optimizer_F.zero_grad()
                self.loss_G = self.compute_G_loss()
                self.loss_R = self.compute_R_loss()
                self.loss_G_total = self.loss_G+self.loss_R
                self.loss_G_total.backward()
                self.optim_R.step()
                self.optim_G.step()
                optimizer_F.step()
                # self.optim_G.zero_grad()
                # fake_B = self.netG(self.real_A)
                # Trans = self.netR(fake_B,self.real_A)
                # SysRegist_AB = self.stn(fake_B,Trans)
                # SR_loss = self.lambda_Corr*self.criterionL1(SysRegist_AB,self.real_B)
                # pred_fake0 = self.netD(fake_B)
                # adv_loss = self.lambda_Adv*self.criterionGAN(pred_fake0,True)
                # SM_loss = self.lambda_smooth*self.smooothing_loss(Trans)
                # total_loss = SM_loss+SR_loss+adv_loss
                # total_loss.backward()
                # self.optim_R.step()
                # self.optim_G.step()
                # self.optim_D.zero_grad()
                # with torch.no_grad():
                #     fake_B = self.netG(self.real_A)
                # pred_fake = self.netD(fake_B)
                # pred_real = self.netD(self.real_B)
                # loss_D = self.lambda_Adv*(self.criterionGAN(pred_fake,False)+self.criterionGAN(pred_real,True))
                # loss_D.backward()
                # self.optim_D.step()
                
                pbar.set_description((
                    f"Epoch:[{epoch+1} / {self.config.training.n_epochs}] "
                    f" iter:{self.global_step} loss_D:{self.loss_D.detach().mean()},loss_G:{self.loss_G_total.detach().mean()}"))
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
                # break
            pbar.close()
            # break
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
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt and self.isTrain else self.real_A
        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        self.Trans = self.netR(self.fake_B,self.real_B)
        self.Reg_B = self.stn(self.fake_B,self.Trans)
    
    def compute_R_loss(self):
        SR_loss = self.lambda_Corr*self.criterionL1(self.Reg_B,self.real_B)
        smooth_loss = self.lambda_smooth*Grad(self.device).loss(self.Trans)
        # print(f"SR_loss:{SR_loss},smooth_loss:{smooth_loss}")
        return smooth_loss+SR_loss
    
    def compute_D_loss(self):
        fake_BRegB=torch.cat((self.real_B,self.Reg_B),1)
        pred_fake = self.netD(fake_BRegB.detach())
        self.loss_D_fakeReg = self.criterionGAN(pred_fake,False)
        fake_AB=torch.cat((self.real_A,self.fake_B),1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake,False)
        real_AB = torch.cat((self.real_A,self.real_B),1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real,True)
        self.loss_D = (self.loss_D_fake+self.loss_D_real+self.loss_D_fakeReg)*0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake_AB = torch.cat((self.real_A,self.fake_B),1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake,True)*self.lambda_Adv
        fake_BRegB = torch.cat((self.real_B,self.Reg_B),1)
        pred_fake_reg = self.netD(fake_BRegB)
        self.loss_G_GAN += self.criterionGAN(pred_fake,True)*self.lambda_Adv
        
        self.loss_G_L1 = self.criterionL1(self.Reg_B,self.real_B)*self.lambda_Cyc
        
        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
        feat_reg_B =  self.netG(self.Reg_B, self.nce_layers, encode_only=True)
        if self.nce_idt:
            feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)
        
        if self.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_reg_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        loss_NCE_all = self.loss_NCE

        if self.nce_idt and self.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE_Y = 0.0
        loss_NCE_all += self.loss_NCE_Y
        
        self.loss_G = loss_NCE_all  + self.loss_G_GAN + self.loss_G_L1
        return self.loss_G
    
    

    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, nce_layers):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.flip_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    # def smooothing_loss(self,y_pred):
    #     dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    #     dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    #     dx = dx*dx
    #     dy = dy*dy
    #     d = torch.mean(dx) + torch.mean(dy)
    #     grad = d 
    #     return d
    
    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 
    
    def resume_training(self, checkpoint_path):
        """
        恢复训练状态，包括模型权重、优化器状态、调度器状态和训练进度。
        """
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.netR.load_state_dict(checkpoint['netR_state_dict'])
        self.optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
        self.optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
        self.optim_R.load_state_dict(checkpoint['optim_R_state_dict'])
        self.lr_schedulerG.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.lr_schedulerD.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.lr_schedulerR.load_state_dict(checkpoint['scheduler_R_state_dict'])
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
            'netR_state_dict': self.netR.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'optim_R_state_dict': self.optim_R.state_dict(),
            'scheduler_G_state_dict': self.lr_schedulerG.state_dict(),
            'scheduler_D_state_dict': self.lr_schedulerD.state_dict(),
            'scheduler_R_state_dict': self.lr_schedulerR.state_dict(),
            'epoch': epoch,
            'step': self.global_step,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")