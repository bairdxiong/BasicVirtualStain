
"""
读取IHC图像的同时还读取其DAB图像/细胞核图像
"""
import os 
from  torch.utils.data import Dataset
import albumentations as A 
import numpy as np 
from PIL import Image
import torchvision.transforms as tt
import torchvision.transforms.functional as TF
from dataset.nuclei_dab_util import *
from Register import Registers

@Registers.datasets.register_with_name("MISTDABDataset")
class PatchAlignedWithDABDataset(Dataset):
    def __init__(self,data_config,phase='train',max_dataset_size=float('inf'),is_blur=True):
        super().__init__()
        self.dataroot=data_config.dataroot
        self.phase = phase
        self.is_blur = is_blur
        subclass = data_config.subclass
        self.image_size = data_config.load_size
        self.crop_size = data_config.crop_size
        assert len(subclass) == 2, f"Exactly 2 subclasses required, got {len(subclass)}"
        assert phase in ['train', 'val', 'test'], f"Invalid phase: {phase}"
        if phase == 'test':
            phase = 'val'
        self.dir_A = os.path.join(self.dataroot,phase+subclass[0])
        self.dir_B = os.path.join(self.dataroot,phase+subclass[1]) # trainB testB
        if phase == 'train':
            self.dir_dab = os.path.join(self.dataroot,phase+"_IHC_dab")
            self.dir_dab_mask = os.path.join(self.dataroot,phase+"_IHC_dab_mask")
            self.dir_nuclei_map = os.path.join(self.dataroot,phase+"_IHC_nuclei_map")
            self.dab_images = sorted(os.listdir(self.dir_dab))
            self.dab_mask_images = sorted(os.listdir(self.dir_dab_mask))
            self.nuclei_maps = sorted(os.listdir(self.dir_nuclei_map))
        # Verify directory existence
        assert os.path.exists(self.dir_A), f"Directory not found: {self.dir_A}"
        assert os.path.exists(self.dir_B), f"Directory not found: {self.dir_B}"

        # Load and sort image lists
        self.A_images = sorted(os.listdir(self.dir_A))
        self.B_images = sorted(os.listdir(self.dir_B))

        # Apply dataset size limit
        self.length = min(len(self.A_images), len(self.B_images), max_dataset_size)


    def __len__(self):
        return self.length
    def __getitem__(self, index):
        # Get file paths while maintaining order correspondence
        
        img_A_name = self.A_images[index]
        img_B_name = self.B_images[index]

        img_A_path = os.path.join(self.dir_A, img_A_name)
        img_B_path = os.path.join(self.dir_B, img_B_name)

        # Load images with error handling
        try:
            image_A = Image.open(img_A_path).convert('RGB')
            image_B = Image.open(img_B_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading images {img_A_path} and {img_B_path}: {str(e)}")
        
        # Validate image dimensions
        if image_A.size != image_B.size:
            raise ValueError(f"Dimension mismatch: {image_A.size} vs {image_B.size}")

        # Convert to tensors and normalize
        train_transform = tt.Compose([    
            tt.Resize([self.image_size,self.image_size]),
            tt.RandomCrop(self.crop_size),
            tt.RandomHorizontalFlip(p=0.5),
            tt.ToTensor(),
            
        ])
        norm_aug = tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        test_transform = tt.Compose([
            tt.Resize([self.image_size,self.image_size]),
            tt.ToTensor(),
            
        ])
        if self.phase=='train':
            transform = train_transform
        else:
            transform = test_transform
        image_A = transform(image_A)
        image_B = transform(image_B)
        image_A = norm_aug(image_A)
        image_B = norm_aug(image_B)
        if self.phase == "train":
            dab_name = self.dab_images[index]
            IHC_DAB = Image.open(os.path.join(self.dir_dab,dab_name)).convert("RGB")
            IHC_DAB = transform(IHC_DAB)
            IHC_DAB = norm_aug(IHC_DAB)
            dab_mask_name = self.dab_mask_images[index]
            dab_mask = Image.open(os.path.join(self.dir_dab_mask,dab_mask_name)).convert("L")
            DAB_mask = transform(dab_mask)
            nuclei_map_name = self.nuclei_maps[index]
            nuclei_map = Image.open(os.path.join(self.dir_nuclei_map,nuclei_map_name)).convert("L")
            nuclei_map = transform(nuclei_map)
            
            return {
                'A': image_A,
                'B': image_B,
                'DAB': IHC_DAB,
                'DAB_Mask': DAB_mask,
                'Nuclei_map': nuclei_map,
                'A_paths': img_A_path,
                'B_paths': img_B_path,
                'index': index
            }
            
        return {
            'A': image_A,
            'B': image_B,
            'A_paths': img_A_path,
            'B_paths': img_B_path,
            'index': index
        }
