# modified from https://github.com/balball/TDKstain/blob/main/preprocess/get_dab_mask.py
import os
import cv2
import copy
import time
import numpy as np
import skimage
import matplotlib.pyplot as plt


def get_dab_mask(ihc_rgb, if_blur=False):
    threshold = 0.15
    
    ihc_hed = skimage.color.rgb2hed(ihc_rgb)
    ihc_dab = ihc_hed[:, :, 2]
    
    null_channal = np.zeros_like(ihc_dab)
    ihc_dab_rgb = skimage.color.hed2rgb(np.stack((null_channal, null_channal, ihc_dab), axis=-1))
    
    ihc_dab_hsv = skimage.color.rgb2hsv(ihc_dab_rgb)
    ihc_dab_s = ihc_dab_hsv[:, :, 1]
    # print(np.max(ihc_dab_s), np.min(ihc_dab_s))
    
    ihc_dab_mask = copy.deepcopy(ihc_dab_s)
    ihc_dab_mask[ihc_dab_mask > threshold] = 255
    ihc_dab_mask[ihc_dab_mask <= threshold] = 0
    
    ihc_dab_rgb = np.array(ihc_dab_rgb * 255, dtype=np.uint8)
    ihc_dab_mask = np.array(ihc_dab_mask, dtype=np.uint8)
    
    if if_blur:
        ihc_dab_mask = cv2.GaussianBlur(ihc_dab_mask, (9,9), 3, 3)
    return ihc_dab_rgb, ihc_dab_mask