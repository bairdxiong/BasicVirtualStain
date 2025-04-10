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
from src.losses import FocalNCELoss,GANLoss,PatchNCELoss,VGGLoss,Gauss_Pyramid_Conv,PatchAlignmentLoss
from src.utils.args_util import dict2namespace
import src.utils.dist_util as dist_util
from src.utils.img_util import visualize_A2B,tensor_to_image
from Runner.GANBased.CUTRunner import CUTRunner


@Registers.runners.register_with_name("PPT_Runner")
class PPTRunner(CUTRunner):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.vgg_layer = 2
        
        # scheduler
        self.n_epochs_decay = config.training.n_epochs_decay
        
        # modified parameters
        self.opt = dict2namespace({
            'weight_norm': '',
            'n_downsampling': 4,
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
        self.netF = net[1]
        self.netD = net[2]
        self.optim_G = optim[0]
        self.optim_D = optim[1]
        self.lr_schedulerG = scheduler[0]
        self.lr_schedulerD = scheduler[1]
        # loss
        self.criterionGAN = GANLoss(self.config.lambda_loss_fn.gan_mode).to(self.device)
        self.criterionNCE = []
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(FocalNCELoss(self.opt).to(self.device))
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionVGG = VGGLoss(self.gpu_ids)
        self.P = Gauss_Pyramid_Conv(num_high=5)
        self.gp_weights = [1.0]*6
        self.criterionMisalignment = PatchAlignmentLoss()
        
        first_step = False

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
                    self.compute_G_loss().backward()                   # calculate graidents for G
                    if self.lambda_NCE>0.0:
                        optimizer_F = get_optimizer(self.config.model.model_G.optimizer,self.netF.parameters())
                    first_step = True
                self.forward()

                # update D
                self.set_requires_grad(self.netD, True)
                self.optim_D.zero_grad()
                self.loss_D = self.compute_D_loss()
                self.loss_D.backward()
                self.optim_D.step()
                # update G
                self.set_requires_grad(self.netD, False)
                self.optim_G.zero_grad()
                optimizer_F.zero_grad()
                self.loss_G = self.compute_G_loss()
                self.loss_G.backward()
                self.optim_G.step()
                optimizer_F.step()

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
    
    def compute_D_loss(self):
        return super().compute_D_loss()
    def forward(self):
        return super().forward()
    
    def compute_G_loss(self):
        fake = self.fake_B

        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
        if self.nce_idt:
            feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

        pred_fake = self.netD(self.fake_B)
        pred_real = self.netD(self.real_B)
        
        self.loss_G_GAN = self.criterionGAN(pred_fake,True).mean()
        
        # contrastive loss
        self.loss_NCE = self.calculate_NCE_loss(feat_real_A,feat_fake_B,self.netF, self.nce_layers)
        self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B,feat_fake_B,self.netF, self.nce_layers)
        self.loss_NCE_Y_hat = self.calculate_NCE_loss(feat_idt_B,feat_real_B,self.netF, self.nce_layers)
        self.loss_contra = self.loss_NCE+self.loss_NCE_Y+self.loss_NCE_Y_hat
        # patch alignment loss
        self.loss_patch_alignment = self.criterionMisalignment(self.real_B,self.fake_B)
        
        # content loss
        self.loss_content = self.criterionVGG(self.fake_B,self.real_B)
        # frequency loss
        p_fake_B = self.P(self.fake_B)
        p_real_B = self.P(self.real_B)
        loss_pyrmid = [self.criterionL1(pf,pr) for pf,pr in zip(p_fake_B,p_real_B)]
        weights = self.gp_weights
        self.loss_freq = torch.mean(torch.stack(loss_pyrmid))
        
        self.loss_G = self.loss_G_GAN+self.loss_freq+self.loss_content+self.loss_patch_alignment+self.loss_contra
        return self.loss_G