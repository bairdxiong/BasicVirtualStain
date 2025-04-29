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
from src.models.sampleF_net import PatchAttnSampleF
from src.models.basic_discriminator import NLayerDiscriminator,PixelDiscriminator
from dataset import get_dataset,get_optimizer
from src.losses import GANLoss,PatchNCELoss,MC_Loss
from src.utils.args_util import dict2namespace
import src.utils.dist_util as dist_util
from src.utils.img_util import visualize_A2B,tensor_to_image
from Runner.GANBased.CUTRunner import CUTRunner


@Registers.runners.register_with_name("SIMGAN_Runner")
class SIMGANRunner(CUTRunner):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.lambda_GAN = 1.0
        self.lambda_NCE = 1.0
        self.lambda_CC = 10.0
        self.eps = 1.0 # epsilon of OT
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
            'flip_equivariance':self.flip_equivariance,
            'gpu_ids': self.gpu_ids,
            'eps': self.eps,
            'cost_type': 'easy', # ot type
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
        self.criterionOT = []
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(self.opt).to(self.device))
        for nce_layer in self.nce_layers:
            self.criterionOT.append(MC_Loss(self.opt).to(self.device))
            
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

        CC_loss = self.calculate_CC_loss(self.real_B, self.fake_B)
        OT_loss = self.calculate_OT_loss(self.real_A, self.real_B, self.fake_B)
        
        self.loss_G = self.loss_G_GAN + loss_NCE_all + OT_loss + CC_loss
        return self.loss_G
    
    def calculate_OT_loss(self, src, tgt, gen):
        n_layers = len(self.nce_layers)
        feat_src = self.netG(src, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_src = [torch.flip(fq, [3]) for fq in feat_src]

        feat_tgt = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_gen = self.netG(gen, self.nce_layers, encode_only=True)
        feat_src_pool, sample_ids = self.netF(feat_src, self.num_patches, None)
        feat_tgt_pool, _ = self.netF(feat_tgt, self.num_patches, sample_ids)
        feat_gen_pool, _ = self.netF(feat_gen, self.num_patches, sample_ids)
        total_ot_loss = 0.0
        for f_src, f_tgt, f_gen, crit, nce_layer in zip(feat_src_pool, feat_tgt_pool, feat_gen_pool, self.criterionOT, self.nce_layers):
            loss = crit(f_src, f_tgt, f_gen) * 10000
            total_ot_loss += loss.mean()

        return total_ot_loss / n_layers

    
    def calculate_CC_loss(self, src, tgt):
        matrix_src = self.cal_matrix(src).detach().to(self.device)
        matrix_tgt = self.cal_matrix(tgt).to(self.device)
        #tensor_src = torch.tensor(matrix_src).to(self.device)
        #tensor_src = torch.tensor([item.cpu().detach().numpy() for item in matrix_src]).to(self.device)
        #tensor_tgt = torch.tensor([item.cpu().detach().numpy() for item in matrix_tgt]).to(self.device)
        #tensor_tgt = torch.tensor(matrix_tgt).to(self.device)
        CC_loss = F.l1_loss(matrix_src, matrix_tgt) * 10
        return CC_loss      
    
    def cal_matrix(self, batch_images):

        cosine_similarity_matrices = torch.zeros(len(self.nce_layers), batch_images.size(0), batch_images.size(0))
        feat = self.netG(batch_images, self.nce_layers, encode_only=True)

        # 遍历每个四维张量
        for idx, tensor in enumerate(feat):
    # 切分为8个四维张量，第一维度为1
            sub_tensors = torch.split(tensor, 1, dim=0)

    # 计算任意两个张量之间的余弦相似度
            for i in range(batch_images.size(0)):
                for j in range(batch_images.size(0)):
                    vector_i = sub_tensors[i].view(-1)
                    vector_j = sub_tensors[j].view(-1)

            # 使用余弦相似度公式计算
                    similarity = F.cosine_similarity(vector_i, vector_j, dim=0)
                    cosine_similarity_matrices[idx, i, j] = similarity
        return cosine_similarity_matrices