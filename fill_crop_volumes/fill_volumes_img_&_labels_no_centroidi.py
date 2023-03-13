# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:23:19 2023

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
    EnsureChannelFirstd,
    AsDiscreted,
    AsDiscrete,
    Activationsd,
    Orientationd,
    Flipd,
    ToTensord

)
from monai.utils import  set_determinism
from monai.networks.nets import SwinUNETR, UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import tensorflow as tf

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import shutil


# directory= 'C:/Users/omobo/Desktop/all image/GAMBE'

# in_images = os.path.join(directory,'imagesTs-gamba')
# in_labels = os.path.join(directory,'labelsTs-gamba')
# # in_preds = os.path.join(directory,'outputsTs-coscia')


# # outputs_directory = "C:/Users/omobo/Desktop/VOLUMI/COSCIA"

# # out_images = os.path.join(outputs_directory,'images')
# # out_labels = os.path.join(outputs_directory,'labels')
# # #out_preds = os.path.join(outputs_directory,'outputs')


# images_path = sorted(glob(os.path.join(in_images,'*.png')))
# labels_path = sorted(glob(os.path.join(in_labels,'*.png')))
# # preds_path = sorted(glob(os.path.join(in_preds,'*.png')))

pred_unet_coscia = "C:/Users/omobo/Desktop/no_zeros/COSCIA 0.8339 e 0.8347"
pred_unet_gamba = "C:/Users/omobo/Desktop/no_zeros/GAMBA 0.8068"

#directory = "C:/Users/omobo/Desktop/Unet-resnet/COSCIA"
# directory = "C:/Users/omobo/Desktop/Unet-resnet/GAMBA"

#in_labels = os.path.join(directory,'labelTs')
# in_preds = os.path.join(directory,'outputTs (post-proc)')

# out_labels = os.path.join(directory,'label_vol')
# out_preds = os.path.join(directory,'pred_vol')

unet_coscia_path = sorted(glob(os.path.join(pred_unet_coscia,'*.png')))
unet_gamba_path = sorted(glob(os.path.join(pred_unet_gamba,'*.png')))

# labels_path = sorted(glob(os.path.join(in_labels,'*.png')))
# preds_path = sorted(glob(os.path.join(in_preds,'*.png')))
#-----------------------------------------------------------------------
lista = []
for i in range(10):
    lista.append(str(i))

def rename (path):
    for i in path:
        image_name = os.path.splitext(i)[0]
        num_slice = image_name.split('_')[-1]
        if num_slice in lista:
                new_i = i.replace('{}.png'.format(num_slice),'0{}.png'.format(num_slice))
                print(new_i)
                os.rename(i,new_i)
                new_mask_name = os.path.basename(new_i)
                print('new mask name : ', new_mask_name)
           
def rename_mask (path):
    for i in path:
        image_name = os.path.splitext(i)[0]
        num_slice = image_name.split('_')[-2]
        if num_slice in lista:
            new_i = i.replace('{}_mask.png'.format(num_slice),'0{}.png'.format(num_slice))
            print(new_i)
            os.rename(i,new_i)
            new_mask_name = os.path.basename(new_i)
            print('new mask name : ', new_mask_name)
                
        else:
            new_i = i.replace('{}_mask.png'.format(num_slice),'{}.png'.format(num_slice))
            print(new_i)
            os.rename(i,new_i)
            new_mask_name = os.path.basename(new_i)
            print('new mask name : ', new_mask_name)
          
def rename_preds (path):
    for i in path:
        image_name = os.path.splitext(i)[0]
        num_slice = image_name.split('_')[-2]
        if num_slice in lista:
            new_i = i.replace('{}_pred.png'.format(num_slice),'0{}.png'.format(num_slice))
            print(new_i)
            os.rename(i,new_i)
            new_mask_name = os.path.basename(new_i)
            print('new mask name : ', new_mask_name)
                
        else:
            new_i = i.replace('{}_pred.png'.format(num_slice),'{}.png'.format(num_slice))
            print(new_i)
            os.rename(i,new_i)
            new_mask_name = os.path.basename(new_i)
            print('new mask name : ', new_mask_name)                     
    
rename(unet_coscia_path)
rename(unet_gamba_path)
#rename_preds(preds_path)

#%%

#ricreare il volume

# images_path = sorted(glob(os.path.join(in_images,'*.png')))
# labels_path = sorted(glob(os.path.join(in_labels,'*.png')))
# preds_path = sorted(glob(os.path.join(in_preds,'*.png')))


# n_slices_0_2 = []
# for i in range(32):
#     n_slices_0_2.append(i)
    
# n_slices_1 = []
# for i in range(72):
#     n_slices_1.append(i)

def make_vol_dir(in_dir):
    path = sorted(glob(os.path.join(in_dir,'*.png')))
    # list_id = []
    # for i in path:
    #     image_name = os.path.basename(i)
    #     image_name = image_name.split('_')
    #     image_name = image_name[0:-1]
    #     image_id = '_'.join(image_name)
    #     if image_id not in list_id:
    #         list_id.append(image_id)
            

    # for i in list_id:
    #     vol_dir = os.path.join(in_dir,'{}'.format(i))
    #     if not os.path.exists(vol_dir):
    #       os.mkdir(vol_dir)
          
    for i in path:
        image_name = os.path.basename(i)
        image_name = image_name.split('_')
        image_name = image_name[0:-1]
        image_id = '_'.join(image_name)
        
        dest = os.path.join(in_dir,'{}'.format(image_id))
        shutil.move(i,dest)

make_vol_dir(pred_unet_coscia)    
make_vol_dir(pred_unet_gamba)    


    
#------------------------------FINE PRIMA PARTE--------------------------------
#%%
"""STEP MANUALE:
    METTERE IN TUTTE LE CARTELLE CHE CONTENGONO LE PATCH I FILE NIFTI DI RIFERIMENTO"""    

#%%
#directory dei volumi di riferimento:
#le immagini 3D sono in histogram matched
#le segmentazioni 3D manuali sono in DATASET 3D flip.

# #--------------------------DEBUG---------------------------------
# in_dir = in_labels
# out_dir = out_labels


#CODICE VECCHIO, VOLUMI DI COSCIA E GAMBA RIPRISTINATI SENZA TENERE CONTO DEL CENTROIDE

def make_volumes(in_dir,out_dir,pad):
    all_dir = sorted(glob(os.path.join(in_dir,'*'))) 
      
        
    for path in all_dir:   #--> path = "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1"
        
        vol_path = glob(os.path.join(path,'*.nii'))  #--> "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1\0_005_1_1_mask.nii"
        vol_name = os.path.basename(vol_path[0])  #-->"0_005_1_1_mask.nii"
        print(vol_name)
        original_volume = sitk.ReadImage(vol_path[0])
        or_vol_array = sitk.GetArrayFromImage(original_volume)
        
        num_slices = or_vol_array.shape[0]  #--> 32
        width = or_vol_array.shape[1]  #-->192
        height = or_vol_array.shape[2] #-->256
        
        images = sorted(glob(os.path.join(path,'*.png')))
        
        volume_array = np.zeros((height,width,num_slices))
        
        
        for n,img in enumerate(images):
            
            a = cv2.imread(img,cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            #plt.imshow(a)
            b = np.pad(a,((pad,pad),(pad,pad))).astype(np.uint8)
            #plt.imshow(b)
            
            x_c = b.shape[1]/2
            y_c = b.shape[0]/2
            ymin = int(y_c - height/2) 
            ymax = int(y_c + height/2) 
            xmin = int(x_c - width/2) 
            xmax = int(x_c + width/2) 
            
            img_patch = (b[ymin:ymax,xmin:xmax]).astype(np.uint8)
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("manual_mask {}".format(os.path.basename(img)))
            plt.imshow(img_patch, cmap="nipy_spectral",interpolation='nearest')
            plt.show()
            #mask_patch = mask_array[ymin:ymax,xmin:xmax]
            
            volume_array[:,:,n] = img_patch
            
        volume_array_T = volume_array.T
        volume_array_T =  volume_array_T.astype(np.uint16)   
        newVolume = sitk.GetImageFromArray(volume_array_T)
        newVolume.CopyInformation(original_volume)
        sitk.WriteImage(newVolume, os.path.join(out_dir,vol_name))


    
    
       
def make_img_volumes(in_dir,out_dir,pad):
    
    all_dir = sorted(glob(os.path.join(in_dir,'*'))) 
      
        
    for path in all_dir:   #--> path = "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1"
        
        vol_path = glob(os.path.join(path,'*.nii'))  #--> "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1\0_005_1_1_mask.nii"
        vol_name = os.path.basename(vol_path[0])  #-->"0_005_1_1_mask.nii"
        print(vol_name)
        original_volume = sitk.ReadImage(vol_path[0],sitk.sitkFloat32)
        or_vol_array = sitk.GetArrayFromImage(original_volume)
        
        num_slices = or_vol_array.shape[0]  #--> 32
        width = or_vol_array.shape[1]  #-->192
        height = or_vol_array.shape[2] #-->256
        
        images = sorted(glob(os.path.join(path,'*.png')))
        
        volume_array = np.zeros((height,width,num_slices))
        
        
        for n,img in enumerate(images):
            
            a = cv2.imread(img,cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            #plt.imshow(a)
            b = np.pad(a,((pad,pad),(pad,pad))).astype(np.uint8)
            #plt.imshow(b)
            
            x_c = b.shape[1]/2
            y_c = b.shape[0]/2
            ymin = int(y_c - height/2) 
            ymax = int(y_c + height/2) 
            xmin = int(x_c - width/2) 
            xmax = int(x_c + width/2) 
            
            img_patch = b[ymin:ymax,xmin:xmax]
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("check image")
            plt.imshow(img_patch, cmap="gray")
            plt.show()
            #mask_patch = mask_array[ymin:ymax,xmin:xmax]
            
            volume_array[:,:,n] = img_patch
            
        volume_array_T = volume_array.T    
        newVolume = sitk.GetImageFromArray(volume_array_T)
        newVolume.CopyInformation(original_volume)
        sitk.WriteImage(newVolume, os.path.join(out_dir,vol_name))
        
#COSCIA        
# make_volumes(in_labels,out_labels,40)        
# #make_img_volumes(in_images, out_images,40)            
# make_volumes(in_preds,out_preds,40)
 
#GAMBA
#make_volumes(in_labels,out_labels,60)        
# make_img_volumes(in_images, out_images,100) 
#make_volumes(in_preds,out_preds,60)     
        
 