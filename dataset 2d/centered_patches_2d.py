# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:24:01 2023

@author: omobo
"""

# IMPORT FUNZIONI UTILI

import numpy as np
import os
from glob import glob
import cv2

from monai.data import  Dataset,  DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    SpatialPadd

)
import matplotlib.pyplot as plt


from skimage.morphology import  area_closing,disk,binary_dilation
from skimage.measure import regionprops


#%% 
"""
Definisco le cartelle utili
"""
img_directory = 'C:/Users/omobo/Desktop/histogram matched'
directory= 'C:/Users/omobo/Desktop/DATASET 3D (flip) R'

patches_dir = 'C:/Users/omobo/Desktop/all image'
if not os.path.exists(patches_dir):
  os.mkdir(patches_dir)
#%% creazione cartelle

patches_dir_coscia = os.path.join(patches_dir,'COSCE')
patches_dir_gamba = os.path.join(patches_dir,'GAMBE')

if not os.path.exists(patches_dir_coscia):
  os.mkdir(patches_dir_coscia)
if not os.path.exists(patches_dir_gamba):
  os.mkdir(patches_dir_gamba)

if not os.path.isdir(os.path.join(patches_dir_coscia,'imagesTr-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'imagesTr-coscia'))
if not os.path.isdir(os.path.join(patches_dir_coscia,'imagesVl-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'imagesVl-coscia'))
if not os.path.isdir(os.path.join(patches_dir_coscia,'imagesTs-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'imagesTs-coscia'))
     
if not os.path.isdir(os.path.join(patches_dir_gamba,'imagesTr-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'imagesTr-gamba'))
if not os.path.isdir(os.path.join(patches_dir_gamba,'imagesVl-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'imagesVl-gamba'))
if not os.path.isdir(os.path.join(patches_dir_gamba,'imagesTs-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'imagesTs-gamba'))   

if not os.path.isdir(os.path.join(patches_dir_coscia,'labelsTr-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'labelsTr-coscia'))
if not os.path.isdir(os.path.join(patches_dir_coscia,'labelsVl-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'labelsVl-coscia'))
if not os.path.isdir(os.path.join(patches_dir_coscia,'labelsTs-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'labelsTs-coscia'))
     
if not os.path.isdir(os.path.join(patches_dir_gamba,'labelsTr-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'labelsTr-gamba'))
if not os.path.isdir(os.path.join(patches_dir_gamba,'labelsVl-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'labelsVl-gamba'))
if not os.path.isdir(os.path.join(patches_dir_gamba,'labelsTs-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'labelsTs-gamba'))  
#%%

img_coscia_TR = 'C:/Users/omobo/Desktop/all image/COSCE/imagesTr-coscia'
mask_coscia_TR = 'C:/Users/omobo/Desktop/all image/COSCE/labelsTr-coscia'
img_coscia_VL = 'C:/Users/omobo/Desktop/all image/COSCE/imagesVl-coscia'
mask_coscia_VL = 'C:/Users/omobo/Desktop/all image/COSCE/labelsVl-coscia'
img_coscia_TS = 'C:/Users/omobo/Desktop/all image/COSCE/imagesTs-coscia'
mask_coscia_TS ='C:/Users/omobo/Desktop/all image/COSCE/labelsTs-coscia'

img_gamba_TR = 'C:/Users/omobo/Desktop/all image/GAMBE/imagesTr-gamba'
mask_gamba_TR = 'C:/Users/omobo/Desktop/all image/GAMBE/labelsTr-gamba'
img_gamba_VL = 'C:/Users/omobo/Desktop/all image/GAMBE/imagesVl-gamba'
mask_gamba_VL = 'C:/Users/omobo/Desktop/all image/GAMBE/labelsVl-gamba'
img_gamba_TS = 'C:/Users/omobo/Desktop/all image/GAMBE/imagesTs-gamba'
mask_gamba_TS ='C:/Users/omobo/Desktop/all image/GAMBE/labelsTs-gamba'


#%% 


train_IMG_path_g = sorted(glob(os.path.join(img_directory,'imagesTr-gamba','*.nii')))
train_MASK_path_g = sorted(glob(os.path.join(directory,'labelsTr-gamba','*.nii')))


val_IMG_path_g = sorted(glob(os.path.join(img_directory,'imagesVl-gamba','*.nii')))
val_MASK_path_g = sorted(glob(os.path.join(directory,'labelsVl-gamba','*.nii')))


test_IMG_path_g = sorted(glob(os.path.join(img_directory,'imagesTs-gamba','*.nii')))
test_MASK_path_g = sorted(glob(os.path.join(directory,'labelsTs-gamba','*.nii')))


train_IMG_path_c = sorted(glob(os.path.join(img_directory,'imagesTr-coscia','*.nii')))
train_MASK_path_c = sorted(glob(os.path.join(directory,'labelsTr-coscia','*.nii')))


val_IMG_path_c = sorted(glob(os.path.join(img_directory,'imagesVl-coscia','*.nii')))
val_MASK_path_c = sorted(glob(os.path.join(directory,'labelsVl-coscia','*.nii')))


test_IMG_path_c = sorted(glob(os.path.join(img_directory,'imagesTs-coscia','*.nii')))
test_MASK_path_c = sorted(glob(os.path.join(directory,'labelsTs-coscia','*.nii')))


train_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path_c,train_MASK_path_c)]
val_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path_c,val_MASK_path_c)]
test_files_c = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path_c,test_MASK_path_c)]

train_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path_g,train_MASK_path_g)]
val_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path_g,val_MASK_path_g)]
test_files_g = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path_g,test_MASK_path_g)]

# transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(keys=["image", "label"], pixdim=(
#             1.3594,1.3594,5.0), mode=("bilinear","nearest")),
#         SpatialPadd(keys=["image", "label"],spatial_size = (250,250,-1))
        
#     ]
# )

#%%
"""
 PREPROCESSING:
     CARICO L'IMMAGINE
     CHANNEL-FIRST 
     SPACING 
     ZEROPADDING 
"""

transforms_c = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0938,1.0938,5.0), mode=("bilinear","nearest")),
        SpatialPadd(keys=["image", "label"],spatial_size = (200,200,-1))
        
    ]
)


transforms_g = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.0,1.0,5.0), mode=("bilinear","nearest")),
        SpatialPadd(keys=["image", "label"],spatial_size = (300,300,-1))
        
    ]
)

#COSCIA   
train_ds_c = Dataset(train_files_c , transforms_c)
train_loader_c = DataLoader( train_ds_c, batch_size=1)

# orig_train_ds = Dataset(train_files_c , orig_transforms)
# orig_train_loader = DataLoader( orig_train_ds, batch_size=1)
   
val_ds_c = Dataset(val_files_c , transforms_c)
val_loader_c = DataLoader( val_ds_c, batch_size=1)
   
test_ds_c = Dataset(test_files_c , transforms_c)
test_loader_c = DataLoader( test_ds_c, batch_size=1)
   
#------------------------------------------------------------------------------
#GAMBA
train_ds_g = Dataset(train_files_g , transforms_g)
train_loader_g = DataLoader( train_ds_g, batch_size=1)

   
val_ds_g = Dataset(val_files_g , transforms_g)
val_loader_g = DataLoader( val_ds_g, batch_size=1)
   
test_ds_g = Dataset(test_files_g , transforms_g)
test_loader_g = DataLoader( test_ds_g, batch_size=1)
#------------------------------------------------------------------------------

# per l'operazione morfologica: elemento disco di raggio 5
footprint = disk(5)


#%% FUNZIONI UTILI

def centered_crop(img_array,mask_array,width,height,x_c,y_c):
    
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("A)",fontsize=30)
    # plt.imshow(img_array, cmap="gray")
    # plt.show()
    
    binary_mask = np.copy(mask_array)
    
    binary_mask[binary_mask>0]=1
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("B)",fontsize=30)
    # plt.imshow(binary_mask, cmap="gray")
    # plt.show()
    filled_mask = binary_dilation(binary_mask,footprint=footprint)
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("C)",fontsize=30)
    # plt.imshow(filled_mask, cmap="gray")
    # plt.show()
    filled_mask = area_closing(filled_mask, area_threshold = 700)
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("D)",fontsize=30)
    # plt.imshow(filled_mask, cmap="gray")
    # plt.show()
    filled_mask = filled_mask.astype(np.uint8)
    properties = regionprops(filled_mask)
    if sum(sum(binary_mask)) > 1000:  #binary mask > 1000
        y,x = properties[0].centroid  #se la maschera non è vuota ricalcola i centroidi della nuova maschera, altrimenti usa quelli di prima
        # fig, ax = plt.subplots()
        # ax.imshow(filled_mask, cmap=plt.cm.gray)
        # ax.plot(x_c, y_c, '.r', markersize=15)
        # plt.show
        x_c.append(x)
        y_c.append(y)
    
    width = width
    height = height
    
    ymin = int(y_c[-1] - width/2)  # è uguale a scrivere int(int(y_c[-1])-width/2)
    ymax = int(y_c[-1] + width/2) 
    xmin = int(x_c[-1] - width/2) 
    xmax = int(x_c[-1] + width/2) 
    
    img_patch = img_array[ymin:ymax,xmin:xmax]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("E)",fontsize=30)
    # plt.imshow(img_patch, cmap="gray")
    # plt.show()
    mask_patch = mask_array[ymin:ymax,xmin:xmax]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(mask_patch, cmap="gray")
    # plt.show()
    
    return img_patch,mask_patch



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


    
def make_slices(loader,img_dir,mask_dir,width,height):
    for batch in loader:
        img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        image_id = os.path.splitext(img_name)[0]
        vol_array = batch['image'][0,0,:,:,:].cpu().numpy()
        vol_mask_array = batch['label'][0,0,:,:,:].cpu().numpy().astype(np.uint8)
        
        #inizializzo le coordinate del centro con il generico centro dell'immagine (per le prime slice del volume che non hanno la segm)
        #per poi prendendere il centroide della porzione che contiene l'oggetto
        x_c = [vol_array.shape[1]/2]
        y_c = [vol_mask_array.shape[0]/2]
        
        perc_1,perc_99 = percentile(vol_array,vol_mask_array)
        
        n_slices = vol_array.shape[2] #--> potevo prendere indifferentemente image_array o mask_array perchè hanno lo stesso numero di slices
        # if not os.path.isdir(os.path.join(img_dir,image_id)):
        #      os.mkdir(os.path.join(img_dir,image_id))
        # if not os.path.isdir(os.path.join(mask_dir,image_id)):
        #      os.mkdir(os.path.join(mask_dir,image_id))
        for n in range(n_slices):
            #decommentare per prendere una slice ogni 3 (training)
            #------------------------------------------------------------------
            #if n == 0 or n % 3 == 0 :
            #------------------------------------------------------------------    
                mask_array = vol_mask_array[:,:,n]
                img_array = vol_array[:,:,n]
                    
                try:
                    img_patch,mask_patch = centered_crop(img_array,mask_array, width, height,x_c,y_c)

                    #normalizzo rispetto al 1° e 99° perc l'immagine croppata
                    slice_adj = imadjust(img_patch,perc_1,perc_99)
                    # plt.figure("check", (12, 6))
                    # plt.subplot(1, 1, 1)
                    # plt.title("check image")
                    # plt.imshow(slice_adj, cmap="gray")
                    # plt.show()
                    cv2.imwrite(os.path.join(img_dir,'{}_{}.png'.format(image_id,n)),slice_adj)
                    cv2.imwrite(os.path.join(mask_dir,'{}_{}.png'.format(image_id,n)),mask_patch)  
                except:
                    print(os.path.join(img_dir,'{}_{}.png'.format(image_id,n)))
                
              
#%% APPLICO LE FUNZIONI PER OTTENERE LE PATCH

#coscia 
make_slices(train_loader_c,img_coscia_TR,mask_coscia_TR,144,144)
make_slices(val_loader_c,img_coscia_VL,mask_coscia_VL,144,144)
make_slices(test_loader_c,img_coscia_TS,mask_coscia_TS,144,144)

#gamba 
make_slices(train_loader_g,img_gamba_TR,mask_gamba_TR,112,112)
make_slices(val_loader_g,img_gamba_VL,mask_gamba_VL,112,112)
make_slices(test_loader_g,img_gamba_TS,mask_gamba_TS,112,112)

