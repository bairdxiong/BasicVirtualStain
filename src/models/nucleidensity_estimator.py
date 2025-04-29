import torch
import functools
import torch.nn as nn     
from src.layers import ResnetBlock
import torch.nn.functional as F

class NucleiDensityMapEstimator(nn.Module):
    def __init__(self, input_nc, output_nc, nef=64, n_blocks=6, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect',opt=None):
        super(NucleiDensityMapEstimator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, nef, kernel_size=7, padding='same', bias=use_bias), norm_layer(nef), nn.ReLU(True)]
        
        for _ in range(n_blocks):
            model += [ResnetBlock(nef, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,opt=opt)]
        
        model += [nn.Conv2d(nef, output_nc, kernel_size=7, padding='same')]
        model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, feats):
        size = feats[0].shape[-2:]
        
        aligned_feats = []
        for i in range(len(feats)):
            aligned_feats.append(F.interpolate(feats[i], size=size, mode='bilinear', align_corners=True))
        fusion_feats = torch.cat(aligned_feats, dim=1)
        
        density = self.model(fusion_feats)

        return density
