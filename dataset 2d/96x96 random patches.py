# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:33:21 2022

@author: omobo
"""



import numpy as np

import os
from glob import glob

import cv2

from monai.data import  Dataset,  DataLoader
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    AsDiscreted,
    AsDiscrete,
    Invertd,
    Activationsd,
    CenterSpatialCropd,
    RandCropByPosNegLabeld

)


from monai.utils import  set_determinism




SEED = 0  
set_determinism(SEED)

directory= 'C:/Users/omobo/Desktop/DATASET 3D (flip)' 

#DECOMMENTARE PER CREARE LE CARTELLE

# patches_dir = 'C:/Users/omobo/Desktop/96x96 random'
# if not os.path.exists(patches_dir):
#   os.mkdir(patches_dir)
#%% creazione cartelle

# patches_dir_coscia = 'C:/Users/omobo/Desktop/96x96 random/COSCE'
# patches_dir_gamba = 'C:/Users/omobo/Desktop/96x96 random/GAMBE'

# if not os.path.exists(patches_dir_coscia):
#   os.mkdir(patches_dir_coscia)
# if not os.path.exists(patches_dir_gamba):
#   os.mkdir(patches_dir_gamba)

# if not os.path.isdir(os.path.join(patches_dir_coscia,'imagesTr-coscia')):
#       os.mkdir(os.path.join(patches_dir_coscia,'imagesTr-coscia'))
# if not os.path.isdir(os.path.join(patches_dir_coscia,'imagesVl-coscia')):
#       os.mkdir(os.path.join(patches_dir_coscia,'imagesVl-coscia'))
# if not os.path.isdir(os.path.join(patches_dir_coscia,'imagesTs-coscia')):
#       os.mkdir(os.path.join(patches_dir_coscia,'imagesTs-coscia'))
     
# if not os.path.isdir(os.path.join(patches_dir_gamba,'imagesTr-gamba')):
#       os.mkdir(os.path.join(patches_dir_gamba,'imagesTr-gamba'))
# if not os.path.isdir(os.path.join(patches_dir_gamba,'imagesVl-gamba')):
#       os.mkdir(os.path.join(patches_dir_gamba,'imagesVl-gamba'))
# if not os.path.isdir(os.path.join(patches_dir_gamba,'imagesTs-gamba')):
#       os.mkdir(os.path.join(patches_dir_gamba,'imagesTs-gamba'))   

# if not os.path.isdir(os.path.join(patches_dir_coscia,'labelsTr-coscia')):
#       os.mkdir(os.path.join(patches_dir_coscia,'labelsTr-coscia'))
# if not os.path.isdir(os.path.join(patches_dir_coscia,'labelsVl-coscia')):
#       os.mkdir(os.path.join(patches_dir_coscia,'labelsVl-coscia'))
# if not os.path.isdir(os.path.join(patches_dir_coscia,'labelsTs-coscia')):
#       os.mkdir(os.path.join(patches_dir_coscia,'labelsTs-coscia'))
     
# if not os.path.isdir(os.path.join(patches_dir_gamba,'labelsTr-gamba')):
#       os.mkdir(os.path.join(patches_dir_gamba,'labelsTr-gamba'))
# if not os.path.isdir(os.path.join(patches_dir_gamba,'labelsVl-gamba')):
#       os.mkdir(os.path.join(patches_dir_gamba,'labelsVl-gamba'))
# if not os.path.isdir(os.path.join(patches_dir_gamba,'labelsTs-gamba')):
#       os.mkdir(os.path.join(patches_dir_gamba,'labelsTs-gamba'))
#%%
img_coscia_TR = 'C:/Users/omobo/Desktop/96x96 random/COSCE/imagesTr-coscia'
mask_coscia_TR = 'C:/Users/omobo/Desktop/96x96 random/COSCE/labelsTr-coscia'
img_coscia_VL = 'C:/Users/omobo/Desktop/96x96 random/COSCE/imagesVl-coscia'
mask_coscia_VL = 'C:/Users/omobo/Desktop/96x96 random/COSCE/labelsVl-coscia'
img_coscia_TS = 'C:/Users/omobo/Desktop/96x96 random/COSCE/imagesTs-coscia'
mask_coscia_TS ='C:/Users/omobo/Desktop/96x96 random/COSCE/labelsTs-coscia'

img_gamba_TR = 'C:/Users/omobo/Desktop/96x96 random/GAMBE/imagesTr-gamba'
mask_gamba_TR = 'C:/Users/omobo/Desktop/96x96 random/GAMBE/labelsTr-gamba'
img_gamba_VL = 'C:/Users/omobo/Desktop/96x96 random/GAMBE/imagesVl-gamba'
mask_gamba_VL = 'C:/Users/omobo/Desktop/96x96 random/GAMBE/labelsVl-gamba'
img_gamba_TS = 'C:/Users/omobo/Desktop/96x96 random/GAMBE/imagesTs-gamba'
mask_gamba_TS ='C:/Users/omobo/Desktop/96x96 random/GAMBE/labelsTs-gamba'
    
train_IMG_path_g = sorted(glob(os.path.join(directory,'imagesTr-gamba','*.nii')))
train_MASK_path_g = sorted(glob(os.path.join(directory,'labelsTr-gamba','*.nii')))


val_IMG_path_g = sorted(glob(os.path.join(directory,'imagesVl-gamba','*.nii')))
val_MASK_path_g = sorted(glob(os.path.join(directory,'labelsVl-gamba','*.nii')))


test_IMG_path_g = sorted(glob(os.path.join(directory,'imagesTs-gamba','*.nii')))
test_MASK_path_g = sorted(glob(os.path.join(directory,'labelsTs-gamba','*.nii')))


train_IMG_path_c = sorted(glob(os.path.join(directory,'imagesTr-coscia','*.nii')))
train_MASK_path_c = sorted(glob(os.path.join(directory,'labelsTr-coscia','*.nii')))


val_IMG_path_c = sorted(glob(os.path.join(directory,'imagesVl-coscia','*.nii')))
val_MASK_path_c = sorted(glob(os.path.join(directory,'labelsVl-coscia','*.nii')))


test_IMG_path_c = sorted(glob(os.path.join(directory,'imagesTs-coscia','*.nii')))
test_MASK_path_c = sorted(glob(os.path.join(directory,'labelsTs-coscia','*.nii')))

train_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path_c,train_MASK_path_c)]
val_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path_c,val_MASK_path_c)]
test_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path_c,test_MASK_path_c)]

train_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path_g,train_MASK_path_g)]
val_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path_g,val_MASK_path_g)]
test_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path_g,test_MASK_path_g)]

#%%
num_samples = 6

transforms_c = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        #Orientationd(keys=["image", "label"], axcodes="LPI"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0938,1.0938,5.0), mode=("bilinear","nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96,96,-1),
            pos=1,
            neg=1,
            num_samples= num_samples,
            image_key="image",
            image_threshold=0,
        ),
    ]
)

transforms_g = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        #Orientationd(keys=["image", "label"], axcodes="LPI"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0,1.0,5.0), mode=("bilinear","nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96,96,-1),
            pos=1,
            neg=1,
            num_samples= num_samples,
            image_key="image",
            image_threshold=0,
        ),
    ]
)

#GAMBA

train_ds_g = Dataset(train_files_g , transforms_g)
train_loader_g = DataLoader( train_ds_g, batch_size=1)

   
val_ds_g = Dataset(val_files_g , transforms_g)
val_loader_g = DataLoader( val_ds_g, batch_size=1)
   
test_ds_g = Dataset(test_files_g , transforms_g)
test_loader_g = DataLoader( test_ds_g, batch_size=1)

#VISUALIZZARE IL CONFRONTO RISPETTO AI VOLUMI ORIGINALI

# orig_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
        
#     ]
# )


# orig_train_ds = Dataset(train_files_c , orig_transforms)
# orig_train_loader = DataLoader( orig_train_ds, batch_size=1)


#COSCIA
   
train_ds_c = Dataset(train_files_c , transforms_c)
train_loader_c = DataLoader( train_ds_c, batch_size=1)

   
val_ds_c = Dataset(val_files_c , transforms_c)
val_loader_c = DataLoader( val_ds_c, batch_size=1)
   
test_ds_c = Dataset(test_files_c , transforms_c)
test_loader_c = DataLoader( test_ds_c, batch_size=1)

#==============================solo check======================================

# for batch_t,batch_o in zip(train_loader_c,orig_train_loader):
    
#     check = batch_o["image"]
#     image = batch_t["image"]  
#     # label = check_train_data["label"][0][0] 
#     plt.figure("check", (12, 6))
#     # plt.subplot(1, num_samples + 1, 1)
#     # plt.title("check image")
#     # plt.imshow(check[0,0,:, :, 30], cmap="gray")
    
#     for i in range(num_samples):
#         # plot the slice [:, :, 30]
        
#         plt.subplot(1, num_samples, i+1)
#         plt.title("Random patch num {}".format(str(i+1)))
#         plt.imshow(image[i,0,:, :, 30], cmap="gray")
#     plt.show()
        
    # # plot the slice [:, :, 30]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 5, 1)
    # plt.title("check image")
    # plt.imshow(check[0,0,:, :, 30], cmap="gray")
    # plt.subplot(1, 5, 2)
    # plt.title("1st Random patch")
    # plt.imshow(image[0,0,:, :, 30], cmap="gray")
    # plt.subplot(1, 5, 3)
    # plt.title("2nd Random patch")
    # plt.imshow(image[1,0,:, :, 30], cmap="gray")
    # plt.subplot(1, 5, 4)
    # plt.title("3rd Random patch")
    # plt.imshow(image[2,0,:, :, 30], cmap="gray")
    # plt.subplot(1, 5, 5)
    # plt.title("4th Random patch")
    # plt.imshow(image[3,0,:, :, 30], cmap="gray")
    # plt.show()
#==============================================================================

#%%FUNZIONI UTILI

    
def percentile(image_array,mask_array):
    
    image_array_1D = image_array[mask_array != 0]   #individuo solo la regione dell'immagine che contiene i muscoli
    #calcolo il 1° e il 99° percentile di questa regione
    perc_1 = np.percentile(image_array_1D,1)
    perc_99 = np.percentile(image_array_1D,99)
    
    return perc_1,perc_99

#equivalent of imadjust in MATLAB
#https://stackoverflow.com/questions/49656244/fast-imadjust-in-opencv-and-python
def imadjust(slice_n,perc_1,perc_99):
    slice_adj = (slice_n - perc_1) * (255 / (perc_99 - perc_1))
    slice_adj = np.clip(slice_adj, 0, 255) # in-place clipping
    return slice_adj


def make_patches(loader, img_dir, mask_dir):
    for batch in loader:
        img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        image_id = os.path.splitext(img_name)[0]
        if not os.path.isdir(os.path.join(img_dir,image_id)):
             os.mkdir(os.path.join(img_dir,image_id))
        if not os.path.isdir(os.path.join(mask_dir,image_id)):
             os.mkdir(os.path.join(mask_dir,image_id))
        for i in range(num_samples):
            image_array = batch['image'][i,0,:,:,:].cpu().numpy()
            mask_array = batch['label'][i,0,:,:,:].cpu().numpy().astype(np.uint8)
            
            perc_1,perc_99 = percentile(image_array,mask_array)
            
            n_slices = image_array.shape[2]
            if not os.path.isdir(os.path.join(img_dir,image_id,str(i+1))):
                 os.mkdir(os.path.join(img_dir,image_id,str(i+1)))
            if not os.path.isdir(os.path.join(mask_dir,image_id,str(i+1))):
                 os.mkdir(os.path.join(mask_dir,image_id,str(i+1)))
            
            for n in range(n_slices):
                if n == 0 or n % 10 == 0 :   #seleziono solo le slices più rappresentative
                    slice_n = image_array[:,:,n]
                    slice_adj = imadjust(slice_n,perc_1,perc_99)
                    cv2.imwrite(os.path.join(img_dir,image_id,str(i+1),'{}.png'.format(n)),slice_adj)
                    cv2.imwrite(os.path.join(mask_dir,image_id,str(i+1),'{}_mask.png'.format(n)),mask_array[:,:,n])  
                
#%%        
make_patches(train_loader_c,img_coscia_TR,mask_coscia_TR)
make_patches(val_loader_c,img_coscia_VL,mask_coscia_VL)
make_patches(test_loader_c,img_coscia_TS,mask_coscia_TS)

make_patches(train_loader_g,img_gamba_TR,mask_gamba_TR)
make_patches(val_loader_g,img_gamba_VL,mask_gamba_VL)
make_patches(test_loader_g,img_gamba_TS,mask_gamba_TS)

    


