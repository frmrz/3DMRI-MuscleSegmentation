# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:18:05 2023

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


input_dir_unet_c = "C:/Users/omobo/Desktop/no_zeros/COSCIA 0.8339 e 0.8347/aiuto"
output_dir_unet_c = "C:/Users/omobo/Desktop/FILL_CENTROIDI/COSCIA/UNET"


input_dir_unet_g ="C:/Users/omobo/Desktop/no_zeros/GAMBA 0.8068/aiuto"
output_dir_unet_g = "C:/Users/omobo/Desktop/FILL_CENTROIDI/GAMBA/UNET"

# per l'operazione morfologica: elemento disco di raggio 5
footprint = disk(5)


def fill_centroidi(input_dir,output_dir):
    all_dir = sorted(glob(os.path.join(input_dir,'*'))) 

    for path in all_dir:   #--> path = "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1"

        #====================VOLUME DI RIFERIMENTO=================================
        vol_path = glob(os.path.join(path,'*.nii'))  #--> "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1\0_005_1_1_mask.nii"
        
        file = [{'label':vol_path}]
        
        #riapplico tutte le trasformazioni che avevo applicato per trovare inizialmente il centroide alla maschera binaria
        #questo centro è il centro dell'attuale patch
        transforms = Compose ([LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Spacingd(keys=[ "label"], pixdim=(
            1.3594,1.3594,5.0), mode=("nearest"))
        #SpatialPadd(keys=["label"],spatial_size = (200,200,-1))
        ])

        ds = Dataset(file, transforms)
        loader = DataLoader(ds, batch_size=1)
        
        vol_trans = first(loader)
        
        #questo mi serve per salvare poi il mìvolume con nibabel-------------------
        affine = vol_trans['label_meta_dict']['affine'][0].numpy()
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
            binary_mask = vol_array[:,:,n]  
          
            binary_mask[binary_mask>0]=1
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("B)",fontsize=30)
            plt.imshow(binary_mask, cmap="gray")
            plt.show()
            filled_mask = binary_dilation(binary_mask,footprint=footprint)
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("C)",fontsize=30)
            plt.imshow(filled_mask, cmap="gray")
            plt.show()
            filled_mask = area_closing(filled_mask, area_threshold = 700)
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("D)",fontsize=30)
            plt.imshow(filled_mask, cmap="gray")
            plt.show()
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
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("before_post_proc_slice: {}".format(os.path.basename(img)))
            plt.imshow(a.astype(np.uint8), cmap="nipy_spectral",interpolation='nearest')
            plt.show()

            
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
            post_proc_slice = post_proc_slice.astype(np.uint8)
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("after_post_proc_slice: {}".format(os.path.basename(img)))
            plt.imshow(post_proc_slice, cmap="nipy_spectral",interpolation='nearest')
            plt.show()
            #------------------------------------------------------------------
            b = np.pad(post_proc_slice,((150,150),(150,150))).astype(np.uint8)
            
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("after post_proc_&_zeropad_slice: {}".format(os.path.basename(img)))
            plt.imshow(b, cmap="nipy_spectral",interpolation='nearest')
            plt.show()
            x_c_patch = b.shape[1]/2
            y_c_patch = b.shape[0]/2
            
            ymin = int(y_c_patch - h1)
            ymax = int(y_c_patch + h2)
            xmin = int(x_c_patch - w1)
            xmax = int(x_c_patch + w2)
            
            slice_n = b[ymin:ymax,xmin:xmax]
            plt.figure("check", (12, 6))
            plt.subplot(1, 1, 1)
            plt.title("after post_proc_&_zeropad_slice: {}".format(os.path.basename(img)))
            plt.imshow(slice_n, cmap="nipy_spectral",interpolation='nearest')
            plt.show()
            
            
            #rispetto al centro della patch (che era il vecchio centroide, aggiungiamo i vari delta: w1,w2,h1,h2
            #e otteniamo la slice iniziale
            
            pred_volume_array[:,:,n] = slice_n
            
        image_name = os.path.basename(path)
        # final_vol = nib.Nifti1Image(pred_volume_array, affine)
        # nib.save(final_vol,os.path.join(output_dir,'{}_pred.nii'.format(image_name)))
        # print('{}_pred.nii Saved!'.format(image_name))


fill_centroidi(input_dir_res_c,output_dir_res_c)
fill_centroidi(input_dir_swin_c,output_dir_swin_c)
fill_centroidi(input_dir_res_g,output_dir_res_g)
fill_centroidi(input_dir_swin_g,output_dir_swin_g)