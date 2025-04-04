"""
modified from:
数据集以文件夹形式存储：例如ANHIR，BCI
|= ANHIR
    |= HE
    |= mas
    |= PAS
    |= PASM
|= BCI
    |=HE
        |=train
        |=test
    |=IHC 
        |=train
        |=test
"""
import os 
from  torch.utils.data import Dataset
import albumentations as A 
import numpy as np 
from PIL import Image
import torchvision.transforms as tt
import torchvision.transforms.functional as TF
from Register import Registers

@Registers.datasets.register_with_name("IHCDataset")
class PatchAlignedDataset(Dataset):
    """
    This dataset class handles loading and preprocessing of paired histopathology
    images from two different staining modalities (e.g., Hematoxylin-Eosin and
    Immunohistochemistry), ensuring spatial alignment between corresponding patches.
    Args:
        dataroot (str): Root directory path containing dataset organization
        phase (str): Dataset phase selector ('train', 'val', or 'test'). Default: 'train'
        subclass (list): List containing exactly two staining modalities (e.g., ['HE', 'IHC'])
        max_dataset_size (int, optional): Maximum number of samples to use from dataset. 
            Default: Infinite (use all available samples)
    Attributes:
        dir_A (str): Path to first staining modality directory
        dir_B (str): Path to second staining modality directory
        length (int): Effective dataset size after applying max_dataset_size constraint
        A_images (list): Sorted list of image filenames for first modality
        B_images (list): Sorted list of image filenames for second modality

    Example:
        >>> dataset = PatchAlignedDataset(
        ...     dataroot='./data',
        ...     phase='train',
        ...     subclass=['HE', 'IHC'],
        ...     max_dataset_size=1000
        ... )
        >>> sample = dataset[0]
        >>> print(sample['A'].shape, sample['B'].shape)
        (3, 256, 256) (3, 256, 256)

    """
    def __init__(self,data_config,phase='train',max_dataset_size=float('inf')):
        super().__init__()
        self.dataroot=data_config.dataroot
        self.phase = phase
        subclass = data_config.subclass
        self.image_size = data_config.load_size
        self.crop_size = data_config.crop_size
        assert len(subclass) == 2, f"Exactly 2 subclasses required, got {len(subclass)}"
        assert phase in ['train', 'val', 'test'], f"Invalid phase: {phase}"
        
        self.dir_A = os.path.join(self.dataroot,subclass[0],phase)
        self.dir_B = os.path.join(self.dataroot,subclass[1],phase)

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
            tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        test_transform = tt.Compose([
            tt.Resize([self.image_size,self.image_size]),
            tt.ToTensor(),
            tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        if self.phase=='train':
            transform = train_transform
        else:
            transform = test_transform
        image_A = transform(image_A)
        image_B = transform(image_B)

        return {
            'A': image_A,
            'B': image_B,
            'A_paths': img_A_path,
            'B_paths': img_B_path,
            'index': index
        }




@Registers.datasets.register_with_name("MISTDataset")
class BreastPatchAlignedDataset(Dataset):
    """
    This dataset class handles loading and preprocessing of paired histopathology
    images from two different staining modalities (e.g., Hematoxylin-Eosin and
    Immunohistochemistry), ensuring spatial alignment between corresponding patches.
    Args:
        dataroot (str): Root directory path containing dataset organization
        phase (str): Dataset phase selector ('train', 'val', or 'test'). Default: 'train'
        subclass (list): List containing exactly two staining modalities (e.g., ['HE', 'IHC'])
        max_dataset_size (int, optional): Maximum number of samples to use from dataset. 
            Default: Infinite (use all available samples)
    Attributes:
        dir_A (str): Path to first staining modality directory
        dir_B (str): Path to second staining modality directory
        length (int): Effective dataset size after applying max_dataset_size constraint
        A_images (list): Sorted list of image filenames for first modality
        B_images (list): Sorted list of image filenames for second modality

    Example:
        >>> dataset = PatchAlignedDataset(
        ...     dataroot='./data',
        ...     phase='train',
        ...     subclass=['A', 'B'],
        ...     max_dataset_size=1000
        ... )
        >>> sample = dataset[0]
        >>> print(sample['A'].shape, sample['B'].shape)
        (3, 256, 256) (3, 256, 256)

    """
    def __init__(self,data_config,phase='train',max_dataset_size=float('inf')):
        super().__init__()
        self.dataroot=data_config.dataroot
        self.phase = phase
        subclass = data_config.subclass
        self.image_size = data_config.load_size
        self.crop_size = data_config.crop_size
        assert len(subclass) == 2, f"Exactly 2 subclasses required, got {len(subclass)}"
        assert phase in ['train', 'val', 'test'], f"Invalid phase: {phase}"
        if phase == 'test':
            phase = 'val'
        self.dir_A = os.path.join(self.dataroot,phase+subclass[0])
        self.dir_B = os.path.join(self.dataroot,phase+subclass[1]) # trainB testB

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
            tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        test_transform = tt.Compose([
            tt.Resize([self.image_size,self.image_size]),
            tt.ToTensor(),
            tt.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        if self.phase=='train':
            transform = train_transform
        else:
            transform = test_transform
        image_A = transform(image_A)
        image_B = transform(image_B)

        return {
            'A': image_A,
            'B': image_B,
            'A_paths': img_A_path,
            'B_paths': img_B_path,
            'index': index
        }