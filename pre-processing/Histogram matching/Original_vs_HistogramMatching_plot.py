# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:55:19 2023

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

#from functions_ import  mean_muscle_metrics, plot_results
#%% plottare gli istogrammi a confronto

input_dir = 'C:/Users/omobo/Desktop/DATASET 3D (flip) R'

in_TR_coscia = os.path.join(input_dir,'imagesTr-coscia')
in_VL_coscia = os.path.join(input_dir,'imagesVl-coscia')
in_TS_coscia = os.path.join(input_dir,'imagesTs-coscia')

in_TR_coscia_la = os.path.join(input_dir,'labelsTr-coscia')
in_VL_coscia_la = os.path.join(input_dir,'labelsVl-coscia')
in_TS_coscia_la = os.path.join(input_dir,'labelsTs-coscia')

in_TR_gamba = os.path.join(input_dir,'imagesTr-gamba')
in_VL_gamba = os.path.join(input_dir,'imagesVl-gamba')
in_TS_gamba = os.path.join(input_dir,'imagesTs-gamba') 

in_TR_gamba_la = os.path.join(input_dir,'labelsTr-gamba')
in_VL_gamba_la = os.path.join(input_dir,'labelsVl-gamba')
in_TS_gamba_la = os.path.join(input_dir,'labelsTs-gamba') 

#--------------------------------DEBUG-----------------------------------------
# img_dir = in_TR_coscia
# mask_dir = in_TR_coscia_la

# volums = os.listdir(img_dir)
# masks = os.listdir(mask_dir)
# hm_dir = "C:/Users/omobo/Desktop/PROVA HM/TUTTE_HM"

# for img,mask in zip(volums, masks):
    
#     print(img)
    
#     source = os.path.join(img_dir,img)
#     source_img = sitk.ReadImage(source,sitk.sitkFloat32)
    
#     hm = os.path.join(hm_dir,img)
#     matched_vol = sitk.ReadImage(hm,sitk.sitkFloat32)
    
#     mask_=sitk.ReadImage(os.path.join(mask_dir,mask))
#     mask_array = sitk.GetArrayFromImage(mask_)
#     mask_array[mask_array > 0.5] = 1
#     # rgsmootherfilter = sitk.SmoothingRecursiveGaussianImageFilter()
#     # rgsmootherfilter.SetSigma(0.1)
#     # rgsmootherfilter.SetNormalizeAcrossScale(True)
#     # rgsmoothedimage  = rgsmootherfilter.Execute(source_img)
    
#     source_array = sitk.GetArrayFromImage(source_img)
#     #source_array = sitk.GetArrayFromImage(rgsmoothedimage)
    
#     #-----------------per plot di confronto img vs hist------------------------
#     source_array_T = source_array.T
#     source_array_flatten=source_array.flatten()
    
#     mask_array_flatten = mask_array.flatten()
#     source_array_nonzeros_index = [i for i,x in enumerate(mask_array_flatten) if (x==1)]
    
#     source_nonzeros = source_array_flatten[source_array_nonzeros_index] #<-- di questa devo fare l'hist


#     #--------------------------------------------------------------------------

#     matched_vol_array = sitk.GetArrayFromImage(matched_vol)
#     matched_vol_array_T = matched_vol_array.T
#     matched_vol_array_flatten = matched_vol_array.flatten()
#     matched_vol_nonzeros = matched_vol_array_flatten[source_array_nonzeros_index] 
    

"""
COSA FA QUESTA FUNZIONE? PERMETTE DI VISUALIZZARE GLI ISTOGRAMMI PRIMA E DOPO HISTOGRAM MATCHING
ho salvato tutti i volumi dopo l'histogram matching in una cartella (hm_dir)
prendo i volumi originali e le maschere per vedere solo l'istogramma della regione di interesse escludendo lo sfondo
estrapolo il nome del volume originale (img) e prendo il corrispettivo nella cartella hm_dir
poi li plotto entrambi con i relativi istogrammi per vedere l'effetto dell'hm
"""

def plot_orig_vs_hm(img_dir,mask_dir):
    volums = os.listdir(img_dir)
    masks = os.listdir(mask_dir)
    hm_dir = "C:/Users/omobo/Desktop/PROVA HM/TUTTE_HM"

    for img,mask in zip(volums, masks):
        
        print(img)
        
        source = os.path.join(img_dir,img)
        source_img = sitk.ReadImage(source,sitk.sitkFloat32)
        
        hm = os.path.join(hm_dir,img)
        matched_vol = sitk.ReadImage(hm,sitk.sitkFloat32)
        
        mask_=sitk.ReadImage(os.path.join(mask_dir,mask))
        mask_array = sitk.GetArrayFromImage(mask_)
        mask_array[mask_array > 0.5] = 1
        # rgsmootherfilter = sitk.SmoothingRecursiveGaussianImageFilter()
        # rgsmootherfilter.SetSigma(0.1)
        # rgsmootherfilter.SetNormalizeAcrossScale(True)
        # rgsmoothedimage  = rgsmootherfilter.Execute(source_img)
        
        source_array = sitk.GetArrayFromImage(source_img)
        #source_array = sitk.GetArrayFromImage(rgsmoothedimage)
        
        #-----------------per plot di confronto img vs hist------------------------
        source_array_T = source_array.T
        source_array_flatten=source_array.flatten()
        
        mask_array_flatten = mask_array.flatten()
        source_array_nonzeros_index = [i for i,x in enumerate(mask_array_flatten) if (x==1)]
        
        source_nonzeros = source_array_flatten[source_array_nonzeros_index] #<-- di questa devo fare l'hist


        #--------------------------------------------------------------------------

        matched_vol_array = sitk.GetArrayFromImage(matched_vol)
        matched_vol_array_T = matched_vol_array.T
        matched_vol_array_flatten = matched_vol_array.flatten()
        matched_vol_nonzeros = matched_vol_array_flatten[source_array_nonzeros_index] 
        

        fig2 = plt.figure(figsize=(20,14))
        fig2.suptitle('{}'.format(img),fontsize=22)
        ax1 = plt.subplot(2,2,1)
        z1_plot = ax1.imshow(source_array_T[:,:,20],cmap='gray')
        ax1.set_title('Original',fontsize=20)
        
        fig2.colorbar(z1_plot,ax=ax1)

        ax2 = plt.subplot(2,2,2)
        ax2.hist(source_nonzeros,bins=256)
        ax2.set_title('Histogram',fontsize=20)
        ax2.set_xlabel('Intensity value',fontsize=16)
        ax2.set_ylabel('# voxels',fontsize=16)


        ax3 = plt.subplot(2,2,3)
        z3_plot = ax3.imshow(matched_vol_array_T[:,:,20],cmap='gray')
        ax3.set_title('After Histogram Matching ',fontsize=20)
        fig2.colorbar(z3_plot,ax=ax3)

        ax4 = plt.subplot(2,2,4)
        ax4.hist(matched_vol_nonzeros,bins=256)
        ax4.set_title('Histogram ',fontsize=20)
        ax4.set_xlabel('Intensity value',fontsize=16)
        ax4.set_ylabel('# voxels',fontsize=16)
        plt.show()


        
#%% APPLICO LA FUNZIONE



#COSCIA
plot_orig_vs_hm(in_TR_coscia,in_TR_coscia_la)
plot_orig_vs_hm(in_VL_coscia,in_VL_coscia_la)
plot_orig_vs_hm(in_TS_coscia,in_TS_coscia_la)

# #GAMBA
plot_orig_vs_hm(in_TR_gamba,in_TR_gamba_la)
plot_orig_vs_hm(in_VL_gamba,in_VL_gamba_la)
plot_orig_vs_hm(in_TS_gamba,in_TS_gamba_la)

