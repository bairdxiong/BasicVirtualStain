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
        netF = PatchSampleF(use_mlp=True,init_type=init_type,init_gain=init_gain,gpu_ids=self.gpu_ids,nc=config.model.model_F.netF_nc)
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
        netF = net[1]
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
                schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_G,lr_lambda=lambda_rule)
            else:
                from types import SimpleNamespace
                del config.model.model_D.lr_scheduler.type
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
            self.criterionNCE.append(PatchNCELoss(self.opt).to(self.device))

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
    
    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt and self.isTrain else self.real_A
        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B

        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
        if self.nce_idt:
            feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

        # First, G(A) should fake the discriminator
        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        loss_NCE_all = self.loss_NCE

        if self.nce_idt and self.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE_Y = 0.0
        loss_NCE_all += self.loss_NCE_Y

        self.loss_G = self.loss_G_GAN + loss_NCE_all 
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