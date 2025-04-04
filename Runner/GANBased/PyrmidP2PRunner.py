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


@Registers.runners.register_with_name("PyrmidP2P_Runner")
class PyrmidPix2PixRunner(GANBaseRunner):
    def __init__(self,config):
        super().__init__(config)
        
        self.config = config
        # modified params
        self.pattern = "L1_L2_L3_L4"
        self.lambda_L1 = 25
        self.weight_L2 = 25
        self.weight_L3 = 25
        self.weight_L4 = 25
        self.opt={
            'weight_norm':None,
            'which_perceptual': '4_2',
            
        }
        self.opt = dict2namespace(self.opt)
        
        # use ddp or not
        if config.training.use_DDP:
            rank = dist.get_rank()
            self.device = rank%torch.cuda.device_count()
        else:
            self.device = dist_util.dev()
            
    def initialize_model(self, config,is_test):
        isTrain = not is_test
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
        
        if isTrain:
            self.model_names.append("D")
            nrom_layer_D = get_norm_layer(norm_type=config.model.model_D.norm)
            netD_name = config.model.model_D.model_name
            input_nc = config.model.model_G.output_nc+config.model.model_G.input_nc # netD input_nc = netG_input_nc 
            if netD_name == "basic":
                # PatchGAN
                netD_A = NLayerDiscriminator(input_nc,config.model.model_D.ndf,n_layers=config.model.model_D.n_layer_D,norm_layer=nrom_layer_D,opt=self.opt)
            elif netD_name == 'pixel':
                netD_A = PixelDiscriminator(input_nc,config.model.model_D.ndf,norm_layer=nrom_layer_D)
            else:
                raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD_name)
            init_type = config.model.model_D.init_type
            init_gain = config.model.model_D.init_gan
            self.netD = init_net(netD_A,init_type,init_gain,gpu_ids=self.gpu_ids)
            return [self.netG,self.netD]
        return [self.netG]
    
    def initialize_optimizer_scheduler(self, net, config,is_test):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        netG = net[0]
        optimizer_G = get_optimizer(config.model.model_G.optimizer,parameters=netG.parameters())
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_G,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.model_G.lr_scheduler))
        if not is_test:
            netD = net[1]
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
        self.isTrain = True
        # create net at gpu_ids[0] GPU
        net,optim,scheduler = self.initialize_model_optimizer_scheduler(self.config,is_test=False)
        self.netG = net[0]
        self.netD = net[1]
        self.optim_G = optim[0]
        self.optim_D = optim[1]
        self.lr_schedulerG = scheduler[0]
        self.lr_schedulerD = scheduler[1]
        # loss
        self.criterionGAN = GANLoss('vanilla').to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        
        for epoch in range(start_epoch,self.config.training.n_epochs):
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
                self.fake_B = self.netG(self.real_A)
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optim_D.zero_grad()     # set D's gradients to zero
                self.backward_D()                # calculate gradients for D
                self.optim_D.step()          # update D's weights
                # update G
                self.set_requires_grad(self.netD, False)  # enable backprop for D
                self.optim_G.zero_grad()     # set D's gradients to zero
                self.backward_G()                # calculate gradients for D
                self.optim_G.step()          # update D's weights
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
                    torch.save(self.netG.state_dict(),os.path.join(self.config.result.ckpt_path,f"netG_A2B_{epoch+1}.pth"))
            torch.save(self.netG.state_dict(),os.path.join(self.config.result.ckpt_path,f"netG_A2B_latest.pth"))
                
    
    def backward_D(self):
        fake_AB = torch.cat((self.real_A,self.fake_B),1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake,False)
        
        real_AB = torch.cat((self.real_A,self.real_B),1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real,True)
        self.loss_D = (self.loss_D_fake+self.loss_D_real)*0.5
        self.loss_D.backward()
    
    def backward_G(self):
        self.loss_G = 0.0
        if 'L1' in self.pattern:
            import kornia
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
            self.loss_G += self.loss_G_L1
            if 'L2' in self.pattern:
                octave1_layer2_fake=kornia.filters.gaussian_blur2d(self.fake_B,(3,3),(1,1))
                octave1_layer3_fake=kornia.filters.gaussian_blur2d(octave1_layer2_fake,(3,3),(1,1))
                octave1_layer4_fake=kornia.filters.gaussian_blur2d(octave1_layer3_fake,(3,3),(1,1))
                octave1_layer5_fake=kornia.filters.gaussian_blur2d(octave1_layer4_fake,(3,3),(1,1))
                octave2_layer1_fake=kornia.filters.blur_pool2d(octave1_layer5_fake, 1, stride=2)
                octave1_layer2_real=kornia.filters.gaussian_blur2d(self.real_B,(3,3),(1,1))
                octave1_layer3_real=kornia.filters.gaussian_blur2d(octave1_layer2_real,(3,3),(1,1))
                octave1_layer4_real=kornia.filters.gaussian_blur2d(octave1_layer3_real,(3,3),(1,1))
                octave1_layer5_real=kornia.filters.gaussian_blur2d(octave1_layer4_real,(3,3),(1,1))
                octave2_layer1_real=kornia.filters.blur_pool2d(octave1_layer5_real, 1, stride=2)
                self.loss_G_L2 = self.criterionL1(octave2_layer1_fake, octave2_layer1_real) * self.weight_L2
                self.loss_G += self.loss_G_L2
                if 'L3' in self.pattern:
                    octave2_layer2_fake=kornia.filters.gaussian_blur2d(octave2_layer1_fake,(3,3),(1,1))
                    octave2_layer3_fake=kornia.filters.gaussian_blur2d(octave2_layer2_fake,(3,3),(1,1))
                    octave2_layer4_fake=kornia.filters.gaussian_blur2d(octave2_layer3_fake,(3,3),(1,1))
                    octave2_layer5_fake=kornia.filters.gaussian_blur2d(octave2_layer4_fake,(3,3),(1,1))
                    octave3_layer1_fake=kornia.filters.blur_pool2d(octave2_layer5_fake, 1, stride=2)
                    octave2_layer2_real=kornia.filters.gaussian_blur2d(octave2_layer1_real,(3,3),(1,1))
                    octave2_layer3_real=kornia.filters.gaussian_blur2d(octave2_layer2_real,(3,3),(1,1))
                    octave2_layer4_real=kornia.filters.gaussian_blur2d(octave2_layer3_real,(3,3),(1,1))
                    octave2_layer5_real=kornia.filters.gaussian_blur2d(octave2_layer4_real,(3,3),(1,1))
                    octave3_layer1_real=kornia.filters.blur_pool2d(octave2_layer5_real, 1, stride=2)
                    self.loss_G_L3 = self.criterionL1(octave3_layer1_fake, octave3_layer1_real) * self.weight_L3
                    self.loss_G += self.loss_G_L3
                    if 'L4' in self.pattern:
                        octave3_layer2_fake=kornia.filters.gaussian_blur2d(octave3_layer1_fake,(3,3),(1,1))
                        octave3_layer3_fake=kornia.filters.gaussian_blur2d(octave3_layer2_fake,(3,3),(1,1))
                        octave3_layer4_fake=kornia.filters.gaussian_blur2d(octave3_layer3_fake,(3,3),(1,1))
                        octave3_layer5_fake=kornia.filters.gaussian_blur2d(octave3_layer4_fake,(3,3),(1,1))
                        octave4_layer1_fake=kornia.filters.blur_pool2d(octave3_layer5_fake, 1, stride=2)
                        octave3_layer2_real=kornia.filters.gaussian_blur2d(octave3_layer1_real,(3,3),(1,1))
                        octave3_layer3_real=kornia.filters.gaussian_blur2d(octave3_layer2_real,(3,3),(1,1))
                        octave3_layer4_real=kornia.filters.gaussian_blur2d(octave3_layer3_real,(3,3),(1,1))
                        octave3_layer5_real=kornia.filters.gaussian_blur2d(octave3_layer4_real,(3,3),(1,1))
                        octave4_layer1_real=kornia.filters.blur_pool2d(octave3_layer5_real, 1, stride=2)
                        self.loss_G_L4 = self.criterionL1(octave4_layer1_fake, octave4_layer1_real) * self.weight_L4
                        self.loss_G += self.loss_G_L4
        self.loss_G.backward()