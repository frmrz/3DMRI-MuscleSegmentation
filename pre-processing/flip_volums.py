# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:23:03 2022

@author: Evelin
"""


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.morphology import remove_small_objects,label, area_closing
from skimage import segmentation

from monai.data import  Dataset,  DataLoader
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    AsDiscreted,
    Invertd,
    SaveImage,
    Activationsd,
    Flipd,
    Spacingd

)
from monai.utils import first, set_determinism
from glob import glob
import torch
from monai.data import  decollate_batch


directory= 'C:/Users/Evelin/OneDrive/Desktop/DATASET 3D (flip)'
flip_dir = 'C:/Users/Evelin/OneDrive/Desktop/Flipped_volums_L_to_R'
#flip_dir = 'C:/Users/Evelin/OneDrive/Desktop/Spacing DATASET 3D'
if not os.path.exists(flip_dir):
  os.mkdir(flip_dir)

flip_dir_coscia = os.path.join(flip_dir,'COSCIA')
if not os.path.exists(flip_dir_coscia):
    os.mkdir(flip_dir_coscia)
    
    
flip_dir_gamba = os.path.join(flip_dir,'GAMBA')
if not os.path.exists(flip_dir_gamba):
    os.mkdir(flip_dir_gamba)
    
out_img_coscia_TR = os.path.join(flip_dir_coscia,'imagesTr-coscia')
if not os.path.exists(out_img_coscia_TR):
    os.mkdir(out_img_coscia_TR)
    
out_mask_coscia_TR = os.path.join(flip_dir_coscia,'labelsTr-coscia')
if not os.path.exists(out_mask_coscia_TR):
    os.mkdir(out_mask_coscia_TR)
    
out_img_coscia_VL = os.path.join(flip_dir_coscia,'imagesVl-coscia')
if not os.path.exists(out_img_coscia_VL):
    os.mkdir(out_img_coscia_VL)
    
out_mask_coscia_VL = os.path.join(flip_dir_coscia,'labelsVl-coscia')
if not os.path.exists(out_mask_coscia_VL):
    os.mkdir(out_mask_coscia_VL)
    
out_img_gamba_TR = os.path.join(flip_dir_gamba,'imagesTr-gamba')
if not os.path.exists(out_img_gamba_TR):
    os.mkdir(out_img_gamba_TR)
    
out_mask_gamba_TR = os.path.join(flip_dir_gamba,'labelsTr-gamba')
if not os.path.exists(out_mask_gamba_TR):
    os.mkdir(out_mask_gamba_TR)
    
out_img_gamba_VL = os.path.join(flip_dir_gamba,'imagesVl-gamba')
if not os.path.exists(out_img_gamba_VL):
    os.mkdir(out_img_gamba_VL)
    
out_mask_gamba_VL = os.path.join(flip_dir_gamba,'labelsVl-gamba')
if not os.path.exists(out_mask_gamba_VL):
    os.mkdir(out_mask_gamba_VL)
    



#CARICO LA MASCHERA SALVATA
#example_filename = os.path.join(directory,'1_001_1_1_L.nii') questo è il primo elemento di train img_path

#---------------------------------------------------------------------------------
train_IMG_path_c = sorted(glob(os.path.join(directory,'imagesTr-coscia','*.nii')))
train_MASK_path_c = sorted(glob(os.path.join(directory,'labelsTr-coscia','*.nii')))


val_IMG_path_c = sorted(glob(os.path.join(directory,'imagesVl-coscia','*.nii')))
val_MASK_path_c = sorted(glob(os.path.join(directory,'labelsVl-coscia','*.nii')))

#---------------------------------------------------------------------------------
train_IMG_path_g = sorted(glob(os.path.join(directory,'imagesTr-gamba','*.nii')))
train_MASK_path_g = sorted(glob(os.path.join(directory,'labelsTr-gamba','*.nii')))


val_IMG_path_g = sorted(glob(os.path.join(directory,'imagesVl-gamba','*.nii')))
val_MASK_path_g = sorted(glob(os.path.join(directory,'labelsVl-gamba','*.nii')))

#---------------------------------------------------------------------------------
train_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path_c,train_MASK_path_c)]
val_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path_c,val_MASK_path_c)]
#test_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path,test_MASK_path)]


train_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path_g,train_MASK_path_g)]
val_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path_g,val_MASK_path_g)]


# transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(keys=["image", "label"], pixdim=(
#             1.3594, 1.3594, 5.0), mode=("bilinear","nearest")),
#         Flipd(keys=["image", "label"], spatial_axis = 0)
#     ]
# )

transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Flipd(keys=["image", "label"], spatial_axis = 0)
    ]
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

train_ds_c = Dataset(train_files_c,transforms)
train_loader_c = DataLoader(train_ds_c,batch_size = 1)

val_ds_c = Dataset(val_files_c,transforms)
val_loader_c = DataLoader(val_ds_c,batch_size = 1)

train_ds_g = Dataset(train_files_g,transforms)
train_loader_g = DataLoader(train_ds_g,batch_size = 1)

val_ds_g = Dataset(val_files_g,transforms)
val_loader_g = DataLoader(val_ds_g,batch_size = 1)


def flip(loader,output_img_dir,output_mask_dir):
    for batch in loader:
        
        img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        img_affine = batch['image_meta_dict']['affine'][0]
        mask_affine = batch['label_meta_dict']['affine'][0]
        img_array = batch['image'][0,0,:,:,:].cpu().numpy() #immagine flippata
        mask_array = batch['label'][0,0,:,:,:].cpu().numpy() #relativa maschera flippata
        
        nib.save(nib.Nifti1Image(img_array, img_affine), os.path.join(output_img_dir,img_name))
        nib.save(nib.Nifti1Image(mask_array, mask_affine), os.path.join(output_mask_dir,'{}_mask.nii'.format(img_name.split('.')[0])))


flip(train_loader_c,out_img_coscia_TR,out_mask_coscia_TR)
flip(val_loader_c,out_img_coscia_VL, out_mask_coscia_VL)
flip(train_loader_g,out_img_gamba_TR,out_mask_gamba_TR)
flip(val_loader_g,out_img_gamba_VL, out_mask_gamba_VL)




# orig_ds = Dataset(train_files,orig_transforms)
# orig_loader = DataLoader(orig_ds,batch_size = 1)

# # FLIPPA?
# check_orig_data = first(orig_loader)
# check_train_data = first(train_loader)
# check_image= check_orig_data["image"][0][0]  #--> no flip
# image = check_train_data["image"][0][0]   #--> sì flip

# # plot the slice [:, :, 30]
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("check image")
# plt.imshow(check_image[:, :, 30], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("trans image")
# plt.imshow(image[:, :, 30], cmap="gray")
# plt.show()

