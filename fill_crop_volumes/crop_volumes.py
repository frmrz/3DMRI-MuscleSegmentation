# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:14:32 2023

@author: omobo
"""



import numpy as np
import os
from glob import glob
from skimage.morphology import remove_small_objects,label, area_closing
from skimage import segmentation
import SimpleITK as sitk

from monai.data import  decollate_batch

import torch


from monai.inferers import sliding_window_inference
from monai.data import  Dataset,  DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    AsDiscreted,
    AsDiscrete,
    Activationsd,
    Orientationd,
    Flipd,
    ToTensord,
    CenterSpatialCropd,

)
from monai.utils import  set_determinism,first
from monai.networks.nets import SwinUNETR, UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib


#tutte le cartelle

main_dir = "C:/Users/omobo/Desktop/VOLUMI/COSCIA - Copia"
#main_dir = "C:/Users/omobo/Desktop/VOLUMI/GAMBA"

# in_imag_path = sorted(glob(os.path.join(main_dir,'images','*.nii')))
# in_label_path = sorted(glob(os.path.join(main_dir,'labels','*.nii')))
# in_resnet_path = sorted(glob(os.path.join(main_dir,'mask_resnet','*.nii')))
# in_swin_path = sorted(glob(os.path.join(main_dir,'mask_swin','*.nii')))
# in_label_3D_path = sorted(glob(os.path.join(main_dir,'labels_3D','*.nii')))
#in_unet_3D_path = sorted(glob(os.path.join(main_dir,'output_unet_3D','*.nii')))
#in_swin_3D_path = sorted(glob(os.path.join(main_dir,'output_swin_3D','*.nii')))
in_unet_3D_path = sorted(glob(os.path.join(main_dir,'imagesTs-coscia','*.nii')))


output_dir = os.path.join(main_dir,'vol_cropped')

# out_imag = os.path.join(output_dir,'images')
# out_label = os.path.join(output_dir,'labels')
# out_resnet = os.path.join(output_dir,'mask_resnet')
# out_swin = os.path.join(output_dir,'mask_swin')
# out_label_3D = os.path.join(output_dir,'labels_3D')
out_unet_3D = os.path.join(output_dir,'imagesTs-coscia')
#out_swin_3D = os.path.join(output_dir,'output_swin_3D')

####per gli altri volumi------------------------------------------------------

# main_dir = 'C:/Users/omobo/Desktop/Unet-resnet/COSCIA'
# #main_dir = "C:/Users/omobo/Desktop/Unet-resnet/GAMBA"

# #in_imag_path = sorted(glob(os.path.join(main_dir,'images','*.nii')))
# in_label_path = sorted(glob(os.path.join(main_dir,'label_vol','*.nii')))
# in_resnet_path = sorted(glob(os.path.join(main_dir,'pred_vol','*.nii')))

# output_dir = os.path.join(main_dir,'vol_cropped')

# #out_imag = os.path.join(output_dir,'images')
# out_label = os.path.join(output_dir,'label_vol')
# out_resnet = os.path.join(output_dir,'pred_vol')

#------------------------------------------------------------

# path = in_imag_path
# out_dir = out_imag

# for file in path:
#     file_name = os.path.basename(file)
#     leg_part = os.path.basename(main_dir)
#     pz_type = file_name.split('_')[0]
    
#     vol = nib.load(file)
#     #print('original_shape =', vol.shape)
    
#     affine = vol.affine
    
#     vol_array = np.array(vol.dataobj)
    
#     if pz_type == '1' and leg_part == 'GAMBA':
#         vol_array = vol_array[:,:,19:56]
#         new_shape = vol_array.shape
#         print('gamba_FSHD_newshape',new_shape)
        
#     elif pz_type == '1' and leg_part == 'COSCIA':
#         vol_array = vol_array[:,:,0:39]     
#         print('new_shape = ',vol_array.shape)
        
#     else:  #--> pz_type = '0' o '2'
#         vol_array = vol_array[:,:,1:26]
#         new_shape = vol_array.shape
        
#     newvol = nib.Nifti1Image ( vol_array, affine )
#     #nib.save(newvol,os.path.join(out_dir,'{}'.format(file_name)))    




def crop_volumes(path,main_dir,out_dir):
    for file in path:
        file_name = os.path.basename(file)
        leg_part = os.path.basename(main_dir)
        pz_type = file_name.split('_')[0]
        
        vol = nib.load(file)
        #print('original_shape =', vol.shape)
        
        affine = vol.affine
        
        vol_array = np.array(vol.dataobj)
        
        if pz_type == '1' and leg_part == 'GAMBA':
            vol_array = vol_array[:,:,23:56]
            new_shape = vol_array.shape
            print('gamba_FSHD_newshape',new_shape)
        elif pz_type == '1' and leg_part == 'COSCIA - Copia':
            vol_array = vol_array[:,:,0:39]     
            print('new_shape = ',vol_array.shape)
            
        else:  #--> pz_type = '0' o '2'
            vol_array = vol_array[:,:,1:26]
            new_shape = vol_array.shape
            
        newvol = nib.Nifti1Image ( vol_array, affine )
        nib.save(newvol,os.path.join(out_dir,'{}'.format(file_name)))
        
        
             
            
# crop_volumes(in_imag_path, main_dir, out_imag)
#crop_volumes(in_label_path, main_dir, out_label)
#crop_volumes(in_resnet_path, main_dir, out_resnet)
# crop_volumes(in_swin_path, main_dir, out_swin)
crop_volumes(in_unet_3D_path, main_dir, out_unet_3D)
#crop_volumes(in_swin_3D_path, main_dir, out_swin_3D)
# crop_volumes(in_label_3D_path, main_dir, out_label_3D)





