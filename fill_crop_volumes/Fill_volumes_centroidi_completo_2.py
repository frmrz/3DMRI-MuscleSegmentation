# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:27:42 2023

@author: omobo
"""



import numpy as np
from numpy import ndarray
import os
from glob import glob
from skimage.morphology import remove_small_objects,label, area_closing
from skimage import segmentation
import os
import SimpleITK as sitk
import nibabel as nib

from monai.data import  decollate_batch

import torch


from monai.inferers import sliding_window_inference
from monai.data import  Dataset,  DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    EnsureChannelFirstd,
    AsDiscreted,
    AsDiscrete,
    Activationsd,
    Orientationd,
    Flipd,
    SpatialPadd,
    Resized,
    ToTensord

)
from monai.utils import  set_determinism
from monai.networks.nets import SwinUNETR, UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import tensorflow as tf

import cv2

import matplotlib.pyplot as plt
import shutil
from monai.utils import first


from skimage.morphology import  area_closing,disk,binary_dilation
from skimage.measure import regionprops



input_dir_res_c ="C:/Users/omobo/Desktop/COSCIA/RESULTS_RESNET/test_all/mask"
input_dir_swin_c = "C:/Users/omobo/Desktop/COSCIA/RESULTS_SWIN/test_all/mask" 


input_dir_res_g = "C:/Users/omobo/Desktop/GAMBA/RESULTS_RESNET/test_all/mask"
input_dir_swin_g = "C:/Users/omobo/Desktop/GAMBA/RESULTS_SWIN/test_all/mask"

output_dir_res_c = "C:/Users/omobo/Desktop/FILL_CENTROIDI/COSCIA/RESNET"
output_dir_swin_c = "C:/Users/omobo/Desktop/FILL_CENTROIDI/COSCIA/SWIN"

output_dir_res_g = "C:/Users/omobo/Desktop/FILL_CENTROIDI/GAMBA/RESNET"
output_dir_swin_g = "C:/Users/omobo/Desktop/FILL_CENTROIDI/GAMBA/SWIN"

input_dir_unet_c = "C:/Users/omobo/Desktop/no_zeros/COSCIA 0.8339 e 0.8347"
output_dir_unet_c = "C:/Users/omobo/Desktop/FILL_CENTROIDI/COSCIA/UNET"


input_dir_unet_g ="C:/Users/omobo/Desktop/no_zeros/GAMBA 0.8068"
output_dir_unet_g = "C:/Users/omobo/Desktop/FILL_CENTROIDI/GAMBA/UNET"

# per l'operazione morfologica: elemento disco di raggio 5
footprint = disk(5)

#---------------------------------------DEBUG------------------------------------

# input_dir = input_dir_unet_g
# output_dir = output_dir_unet_g

# all_dir = sorted(glob(os.path.join(input_dir,'*'))) 

# for path in all_dir:   #--> path = "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1"

#     #====================VOLUME DI RIFERIMENTO=================================
#     vol_path = glob(os.path.join(path,'*.nii'))  #--> "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1\0_005_1_1_mask.nii"
    
#     file = [{'label':vol_path}]
    
#     #riapplico tutte le trasformazioni che avevo applicato per trovare inizialmente il centroide alla maschera binaria
#     #(questo centro è il centro dell'attuale patch)
#     transforms = Compose ([LoadImaged(keys=["label"]),
#     EnsureChannelFirstd(keys=["label"]),
#     Spacingd(keys=[ "label"], pixdim=(
#         1.3594,1.3594,5.0), mode=("nearest"))
#     #SpatialPadd(keys=["label"],spatial_size = (200,200,-1))
#     ])

#     ds = Dataset(file, transforms)
#     loader = DataLoader(ds, batch_size=1)
    
#     vol_trans = first(loader)
    
#     #questo mi serve per salvare poi il mìvolume con nibabel-------------------
#     affine = vol_trans['label_meta_dict']['original_affine'][0].numpy()
#     #--------------------------------------------------------------------------
    
#     vol_array = vol_trans['label'][0,0,:,:,:].cpu().numpy().astype(np.uint8)
    
#     x_c = [vol_array.shape[1]/2]
#     y_c = [vol_array.shape[0]/2]   #se da problemi metti int
    
#     height = vol_array.shape[0] #-->256
#     width = vol_array.shape[1]  #-->192
#     num_slices = vol_array.shape[2]  #--> 32
    
#     #==========================================================================
    
#     images = sorted(glob(os.path.join(path,'*.png')))
    
#     pred_volume_array = np.zeros((height,width,num_slices))
    
#     for n,img in enumerate(images):
        
#        #    QUESTO SERVE PER TROVARE I CENTROIDI, UGUALI A QUELLI RISPETTO AI QUALI SONO STATI OTTENUTE LE PATCH 
#         mask = vol_array[:,:,n] 
#         binary_mask = mask
      
#         binary_mask[binary_mask>0]=1
#         plt.figure("check", (12, 6))
#         plt.subplot(1, 1, 1)
#         plt.title("B)",fontsize=30)
#         plt.imshow(binary_mask, cmap="gray")
#         plt.show()
#         filled_mask = binary_dilation(binary_mask,footprint=footprint)
#         plt.figure("check", (12, 6))
#         plt.subplot(1, 1, 1)
#         plt.title("C)",fontsize=30)
#         plt.imshow(filled_mask, cmap="gray")
#         plt.show()
#         filled_mask = area_closing(filled_mask, area_threshold = 700)
#         plt.figure("check", (12, 6))
#         plt.subplot(1, 1, 1)
#         plt.title("D)",fontsize=30)
#         plt.imshow(filled_mask, cmap="gray")
#         plt.show()
#         filled_mask = filled_mask.astype(np.uint8)
#         properties = regionprops(filled_mask)
#         if sum(sum(binary_mask)) > 1500:
#             y,x = properties[0].centroid  #se la maschera non è vuota ricalcola i centroidi della nuova maschera, altrimenti usa quelli di prima
#             # fig, ax = plt.subplots()
#             # ax.imshow(filled_mask, cmap=plt.cm.gray)
#             # ax.plot(x_c, y_c, '.r', markersize=15)
#             # plt.show
#             x_c.append(x)
#             y_c.append(y)
        
#         x_c_new = x_c[-1]
#         y_c_new = y_c[-1]
        
#         #definisco come fare zeropadding rispetto al centroide
        
#         w1 = x_c_new
#         w2 = (width - x_c_new)
#         h1 = y_c_new 
#         h2 = (height - y_c_new)
        
#         #------------------------------------------------------------------
#         #leggo la segmentazione automatica 2D
#         a = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
       

        
#         data_array = a.astype(np.uint8)
#         #------------------------------------------------------------------                    
#         #remove small objects
#         array = label(data_array)
#         obj_removed = remove_small_objects (array , min_size = 60) 
#         object_removed = segmentation.watershed(
#                   obj_removed, data_array, mask= obj_removed)
        
#         # fill holes
#         post_proc_slice = area_closing(object_removed, area_threshold = 40)   
#         #post_proc_slice è int32
        
#         #------------------------------------------------------------------
#         b = np.pad(post_proc_slice,((150,150),(150,150))).astype(np.uint8)
        
        
#         x_c_patch = b.shape[1]/2
#         y_c_patch = b.shape[0]/2
        
#         ymin = int(y_c_patch - h1)
#         ymax = int(y_c_patch + h2)
#         xmin = int(x_c_patch - w1)
#         xmax = int(x_c_patch + w2)
        
#         slice_n = b[ymin:ymax,xmin:xmax]
        
        
#         #rispetto al centro della patch (che era il vecchio centroide, aggiungiamo i vari delta: w1,w2,h1,h2
#         #e otteniamo la slice iniziale
        
#         pred_volume_array[:,:,n] = slice_n
        
#     image_name = os.path.basename(path)
#     final_vol = nib.Nifti1Image(pred_volume_array, affine)
#     nib.save(final_vol,os.path.join(output_dir,'{}_pred.nii'.format(image_name)))
#     print('{}_pred.nii Saved!'.format(image_name))


def fill_centroidi(input_dir,output_dir):
    all_dir = sorted(glob(os.path.join(input_dir,'*'))) 

    for path in all_dir:   #--> path = "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1"

        #====================VOLUME DI RIFERIMENTO=================================
        vol_path = glob(os.path.join(path,'*.nii'))  #--> "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1\0_005_1_1_mask.nii"
        
        file = [{'label':vol_path}]
        
        #riapplico tutte le trasformazioni che avevo applicato per trovare inizialmente il centroide alla maschera binaria
        #(questo centro è il centro dell'attuale patch)
        transforms = Compose ([LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Spacingd(keys=[ "label"], pixdim=(
            1.0938,1.0938,5.0), mode=("nearest"))
        #SpatialPadd(keys=["label"],spatial_size = (200,200,-1))
        ])
        
        
        #--------------------GAMBA--------------------------------------------
        # transforms = Compose ([LoadImaged(keys=["label"]),
        # EnsureChannelFirstd(keys=["label"]),
        # Spacingd(keys=[ "label"], pixdim=(
        #     1.0,1.0,5.0), mode=("nearest"))
        # #SpatialPadd(keys=["label"],spatial_size = (200,200,-1))
        # ])


        ds = Dataset(file, transforms)
        loader = DataLoader(ds, batch_size=1)
        
        vol_trans = first(loader)
        
        #questo mi serve per salvare poi il mìvolume con nibabel-------------------
        affine = vol_trans['label_meta_dict']['original_affine'][0].numpy()
        #--------------------------------------------------------------------------
        
        vol_array = vol_trans['label'][0,0,:,:,:].cpu().numpy().astype(np.uint8)
        
        x_c = [vol_array.shape[1]/2]
        y_c = [vol_array.shape[0]/2]   #se da problemi metti int
        
        height = vol_array.shape[0] #-->256
        width = vol_array.shape[1]  #-->192
        num_slices = vol_array.shape[2]  #--> 32
        
        #==========================================================================
        
        images = sorted(glob(os.path.join(path,'*.png')))
        
        pred_volume_array = np.zeros((height,width,num_slices))
        
        for n,img in enumerate(images):
            
           #    QUESTO SERVE PER TROVARE I CENTROIDI, UGUALI A QUELLI RISPETTO AI QUALI SONO STATI OTTENUTE LE PATCH 
            mask = vol_array[:,:,n] 
            binary_mask = mask
          
            binary_mask[binary_mask>0]=1
            
            filled_mask = binary_dilation(binary_mask,footprint=footprint)
            
            filled_mask = area_closing(filled_mask, area_threshold = 700)
            
            filled_mask = filled_mask.astype(np.uint8)
            properties = regionprops(filled_mask)
            if sum(sum(binary_mask)) > 1000:
                y,x = properties[0].centroid  #se la maschera non è vuota ricalcola i centroidi della nuova maschera, altrimenti usa quelli di prima
                # fig, ax = plt.subplots()
                # ax.imshow(filled_mask, cmap=plt.cm.gray)
                # ax.plot(x_c, y_c, '.r', markersize=15)
                # plt.show
                x_c.append(x)
                y_c.append(y)
            
            x_c_new = x_c[-1]
            y_c_new = y_c[-1]
            
            #definisco come fare zeropadding rispetto al centroide
            
            w1 = x_c_new
            w2 = (width - x_c_new)
            h1 = y_c_new 
            h2 = (height - y_c_new)
            
            #------------------------------------------------------------------
            #leggo la segmentazione automatica 2D
            a = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
           

            
            data_array = a.astype(np.uint8)
            #------------------------------------------------------------------                    
            #remove small objects
            array = label(data_array)
            obj_removed = remove_small_objects (array , min_size = 60) 
            object_removed = segmentation.watershed(
                      obj_removed, data_array, mask= obj_removed)
            
            # fill holes
            post_proc_slice = area_closing(object_removed, area_threshold = 40)   
            #post_proc_slice è int32
            
            #------------------------------------------------------------------
            b = np.pad(post_proc_slice,((150,150),(150,150))).astype(np.uint8)
            
            
            x_c_patch = b.shape[1]/2
            y_c_patch = b.shape[0]/2
            
            ymin = int(y_c_patch - h1)
            ymax = int(y_c_patch + h2)
            xmin = int(x_c_patch - w1)
            xmax = int(x_c_patch + w2)
            
            slice_n = b[ymin:ymax,xmin:xmax]
            
            
            #rispetto al centro della patch (che era il vecchio centroide, aggiungiamo i vari delta: w1,w2,h1,h2
            #e otteniamo la slice iniziale
            
            pred_volume_array[:,:,n] = slice_n
            
        image_name = os.path.basename(path)
        final_vol = nib.Nifti1Image(pred_volume_array, affine)
        nib.save(final_vol,os.path.join(output_dir,'{}_pred.nii'.format(image_name)))
        print('{}_pred.nii Saved!'.format(image_name))


#fill_centroidi(input_dir_unet_c,output_dir_unet_c)
fill_centroidi(input_dir_swin_c,output_dir_swin_c)
# fill_centroidi(input_dir_res_g,output_dir_res_g)
# fill_centroidi(input_dir_swin_g,output_dir_swin_g)

#fill_centroidi(input_dir_unet_g,output_dir_unet_g)
#%%
#RIPRISTINO LO SPACING CORRETTO

def restore_spacing(output_dir):
    
    print('----------------------------\nRestore spacing')
    label_resample_path = sorted(glob(os.path.join(output_dir,'*.nii')))

    label_resample_path_FSHD =[]
    label_resample_path_MD1HV = []

    for path in label_resample_path:
        image_name = os.path.basename(path)
        pz_id = image_name.split('_')[0]
        if pz_id == '1':
            label_resample_path_FSHD.append(path)
        else:
            label_resample_path_MD1HV.append(path)
            
            
    #files_f =  [{"fshd": fshd_name} for fshd_name in label_resample_path_FSHD]   
    files_o =  [{"other": other_name} for other_name in label_resample_path_MD1HV]   

    # transforms_fshd = Compose(
    #     [
    #         LoadImaged(keys=["fshd"]),
    #         EnsureChannelFirstd(keys=["fshd"]),
    #         Spacingd(keys=["fshd"], pixdim=(
    #             1.359375,1.359375,5.0), mode=("nearest"))
    #     ]
    # )

    transforms_other = Compose(
        [
            LoadImaged(keys=[ "other"]),
            EnsureChannelFirstd(keys=["other"]),
            Spacingd(keys=["other"], pixdim=(
                1.0,1.0,5.0), mode=("nearest")),
            Resized(keys = ["other"], spatial_size = (256,192,-1),mode = 'nearest')
        ]
    )

#

    # ds_fshd = Dataset(files_f ,transforms_fshd)
    # loader_fshd = DataLoader( ds_fshd, batch_size=1)


    ds_other = Dataset(files_o ,transforms_other)
    loader_other = DataLoader( ds_other, batch_size=1)


    # for batch in loader_fshd:
    #     affine= batch['fshd_meta_dict']['affine'][0].numpy()
        
    #     image_path = batch['fshd_meta_dict']['filename_or_obj'][0]
        
    #     image_name = os.path.basename(image_path)
    #     vol_array = batch['fshd'][0,0,:,:,:].cpu().numpy().astype(np.uint8)
    #     final_vol = nib.Nifti1Image(vol_array, affine)
    #     nib.save(final_vol,os.path.join(output_dir_res_c,image_name))
    #     print('{} Saved!'.format(image_name))


    for batch in loader_other:
        affine= batch['other_meta_dict']['affine'][0].numpy()
        
        image_path = batch['other_meta_dict']['filename_or_obj'][0]
        image_name = os.path.basename(image_path)
        vol_array = batch['other'][0,0,:,:,:].cpu().numpy().astype(np.uint8)
        final_vol = nib.Nifti1Image(vol_array, affine)
        nib.save(final_vol,os.path.join(output_dir,image_name))
        
        print('{} Saved!'.format(image_name))
     
        
# restore_spacing(output_dir_res_c)
restore_spacing(output_dir_swin_c)
# restore_spacing(output_dir_res_g)
# restore_spacing(output_dir_swin_g)

#restore_spacing(output_dir_unet_c)

#restore_spacing(output_dir_unet_g)


#%%
#CROP VOLUMES


main_dir_g = "C:/Users/omobo/Desktop/FILL_CENTROIDI/GAMBA"
main_dir_c = "C:/Users/omobo/Desktop/FILL_CENTROIDI/COSCIA"

in_resnet_path_c = sorted(glob(os.path.join(main_dir_c,'SWIN','*.nii')))
in_resnet_path_g = sorted(glob(os.path.join(main_dir_g,'SWIN','*.nii')))



output_dir_c = os.path.join(main_dir_c,'vol_cropped')
if not os.path.exists(output_dir_c):
   os.mkdir(output_dir_c)
   
out_resnet_c = os.path.join(output_dir_c,'SWIN')
if not os.path.exists(out_resnet_c):
   os.mkdir(out_resnet_c)


output_dir_g = os.path.join(main_dir_g,'vol_cropped')
if not os.path.exists(output_dir_g):
   os.mkdir(output_dir_g)
   
out_resnet_g = os.path.join(output_dir_g,'SWIN')
if not os.path.exists(out_resnet_g):
   os.mkdir(out_resnet_g)
   
   
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
        elif pz_type == '1' and leg_part == 'COSCIA':
            vol_array = vol_array[:,:,0:39]     
            print('new_shape = ',vol_array.shape)
            
        else:  #--> pz_type = '0' o '2'
            vol_array = vol_array[:,:,1:26]
            new_shape = vol_array.shape
            
        newvol = nib.Nifti1Image ( vol_array, affine )
        nib.save(newvol,os.path.join(out_dir,'{}'.format(file_name)))
        
        
crop_volumes(in_resnet_path_c, main_dir_c, out_resnet_c)

#crop_volumes(in_resnet_path_g, main_dir_g, out_resnet_g)

