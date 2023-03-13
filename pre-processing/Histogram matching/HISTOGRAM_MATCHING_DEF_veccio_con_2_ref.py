# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:32:39 2023

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
#%% CALCOLARE L'IMMAGINE MEDIA CHE USERO' COME RIFERIMENTO PER L'HOSTOGRAM MATCHING

# eseguo l'istogram matching su due sottogruppi dei nostri volumi, organizzati in base alle loro dimensioni

#STEPS:
    

directory_FSHD_COSCIA = 'C:/Users/omobo/Desktop/PROVA HM/FSHD (160,200,72)/COSCIA'
directory_FSHD_GAMBA = 'C:/Users/omobo/Desktop/PROVA HM/FSHD (160,200,72)/GAMBA'

directory_COSCIA = 'C:/Users/omobo/Desktop/PROVA HM/HV+MD1 (192,256,32)/COSCIA'
directory_GAMBA = 'C:/Users/omobo/Desktop/PROVA HM/HV+MD1 (192,256,32)/GAMBA'


def average_ref(directory):
    IMG_path = sorted(glob(os.path.join(directory,'*.nii')))
    files = [{"image": image_name} for image_name in IMG_path]
    
    transforms = Compose([LoadImaged(keys='image')])

    ds = Dataset(files ,transforms)
    loader = DataLoader(ds, batch_size=len(files))

    all_volumes_batch = first(loader)
    all_volumes_array = all_volumes_batch['image'].cpu().numpy()
    all_volumes_mean = np.mean(all_volumes_array, axis = 0)

    all_volumes_mean_T = all_volumes_mean.T
    
    return all_volumes_mean_T

#%% VOLUMI DI RIFERIMENTO PER I 4 CASI
#questi volumi di riferimento saranno utilizzati per fare istogram matching
ref_FSHD_COSCIA = average_ref(directory_FSHD_COSCIA)

ref_FSHD_GAMBA = average_ref(directory_FSHD_GAMBA)

ref_COSCIA = average_ref(directory_COSCIA)

ref_GAMBA = average_ref(directory_GAMBA)
#%% definisco le cartelle utili
input_dir = 'C:/Users/omobo/Desktop/DATASET 3D (flip) R'

in_TR_coscia = os.path.join(input_dir,'imagesTr-coscia')
in_VL_coscia = os.path.join(input_dir,'imagesVl-coscia')
in_TS_coscia = os.path.join(input_dir,'imagesTs-coscia')

in_TR_gamba = os.path.join(input_dir,'imagesTr-gamba')
in_VL_gamba = os.path.join(input_dir,'imagesVl-gamba')
in_TS_gamba = os.path.join(input_dir,'imagesTs-gamba') 

output_dir = 'C:/Users/omobo/Desktop/histogram matched'

out_TR_coscia = os.path.join(output_dir,'imagesTr-coscia')
out_VL_coscia = os.path.join(output_dir,'imagesVl-coscia')
out_TS_coscia = os.path.join(output_dir,'imagesTs-coscia')

out_TR_gamba = os.path.join(output_dir,'imagesTr-gamba')
out_VL_gamba = os.path.join(output_dir,'imagesVl-gamba')
out_TS_gamba = os.path.join(output_dir,'imagesTs-gamba')

#%%
#PROVA DEBUG

# img_dir = in_TS_coscia
# ref_FSHD = ref_FSHD_COSCIA 
# ref = ref_COSCIA
# output_path = out_TS_coscia
ref_FSHD_COSCIA_flatten = ref_FSHD_COSCIA.flatten()

ref_FSHD_GAMBA_flatten = ref_FSHD_GAMBA.flatten()

ref_COSCIA_flatten = ref_COSCIA.flatten()

ref_GAMBA_flatten = ref_GAMBA.flatten()

# fig1 = plt.figure(figsize=(30,10))
# fig1.suptitle('HISTOGRAMMI DI RIFERIMENTO GRUPPO FSHD',x=0.5)
# ax1 = plt.subplot(1,3,1)
# ax1.hist(ref_FSHD_COSCIA_flatten,bins=256)
# ax1.set_title('COSCIA')

# ax2 = plt.subplot(1,3,2)
# ax2.hist(ref_FSHD_GAMBA_flatten,bins=256)
# ax2.set_title('GAMBA')

# plt.show()


# fig2 = plt.figure(figsize=(30,10))
# fig2.suptitle('HISTOGRAMMI DI RIFERIMENTO GRUPPO HV + MD1',x=0.5)
# ax1 = plt.subplot(1,3,1)
# ax1.hist(ref_COSCIA_flatten,bins=256)
# ax1.set_title('COSCIA')

# ax2 = plt.subplot(1,3,2)
# ax2.hist(ref_GAMBA_flatten,bins=256)
# ax2.set_title('GAMBA')

# plt.show()


def save_histogram_match(img_dir,ref_FSHD,ref,output_path):
    volums = os.listdir(img_dir)

    for img in volums:
        
        print(img)
        source = os.path.join(img_dir,img)
        source_img = sitk.ReadImage(source,sitk.sitkFloat32)
        source_array = sitk.GetArrayFromImage(source_img)
        
        #-----------------per plot di confronto img vs hist------------------------
        source_array_T = source_array.T
        source_array_flatten=source_array.flatten()
        #--------------------------------------------------------------------------
        
        pz_type = img.split('_')[0]
        
        if pz_type == '1':
            
            ref_array = ref_FSHD
        else:
            ref_array = ref
            
        ref_array_flatten = ref_array.flatten()
        
        #---------------------decommenta per visualizzare----------------------
        # fig1 = plt.figure(figsize=(20,10))
        # fig1.suptitle('{}'.format(img))
        # ax1 = plt.subplot(1,3,1)
        # z1_plot = ax1.imshow(source_array_T[:,:,20],cmap='gray')
        # ax1.set_title('Original')
        # fig1.colorbar(z1_plot,ax=ax1)

        # ax2 = plt.subplot(1,3,2)
        # ax2.hist(source_array_flatten,bins=256)
        # ax2.set_title('Histogram')

        # ax3 = plt.subplot(1,3,3)
        # ax3.hist(ref_array_flatten,bins=256)
        # ax3.set_title('Reference Histogram ')
        # plt.show()
        #--------------------------------------------------------------------------
        #                       HISTOGRAM MATCHING    
        #--------------------------------------------------------------------------
        #source_img = souce_itk
        ref_img = sitk.GetImageFromArray(ref_array)


        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(ref_img.GetPixelID())

        histogram_levels = 256
        match_points = 95   #i quantili da considerare. #non c'è molta differenza tra 95 e 100


        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(histogram_levels)
        matcher.SetNumberOfMatchPoints(match_points)
        matcher.SetThresholdAtMeanIntensity(True)
        matched_vol = matcher.Execute(source_img,ref_img)


        matched_vol.CopyInformation(source_img)
        sitk.WriteImage(matched_vol, os.path.join(output_path,img))
        print("########Saved#########")
        
        #visualizzo l'istogramma dell'immagine prima e dopo histogram matching

        matched_vol_array = sitk.GetArrayFromImage(matched_vol)
        matched_vol_array_T = matched_vol_array.T
        matched_vol_array_flatten = matched_vol_array.flatten()

        # fig2 = plt.figure(figsize=(20,10))
        # fig2.suptitle('{}'.format(img))
        # ax1 = plt.subplot(2,2,1)
        # z1_plot = ax1.imshow(source_array_T[:,:,20],cmap='gray')
        # ax1.set_title('Original')
        # fig2.colorbar(z1_plot,ax=ax1)

        # ax2 = plt.subplot(2,2,2)
        # ax2.hist(source_array_flatten,bins=256)
        # ax2.set_title('Histogram')


        # ax3 = plt.subplot(2,2,3)
        # z3_plot = ax3.imshow(matched_vol_array_T[:,:,20],cmap='gray')
        # ax3.set_title('After Histogram Matching ')
        # fig2.colorbar(z3_plot,ax=ax3)

        # ax4 = plt.subplot(2,2,4)
        # ax4.hist(matched_vol_array_flatten,bins=256)
        # ax4.set_title('Histogram ')
        # plt.show()


        
#%% APPLICO LA FUNZIONE

#save_histogram_match(img_dir,ref_FSHD,ref,output_path)

#COSCIA
save_histogram_match(in_TR_coscia,ref_FSHD_COSCIA,ref_COSCIA,out_TR_coscia)
save_histogram_match(in_VL_coscia,ref_FSHD_COSCIA,ref_COSCIA,out_VL_coscia)
save_histogram_match(in_TS_coscia,ref_FSHD_COSCIA,ref_COSCIA,out_TS_coscia)

# #GAMBA
save_histogram_match(in_TR_gamba,ref_FSHD_GAMBA,ref_GAMBA,out_TR_gamba)
save_histogram_match(in_VL_gamba,ref_FSHD_GAMBA,ref_GAMBA,out_VL_gamba)
save_histogram_match(in_TS_gamba,ref_FSHD_GAMBA,ref_GAMBA,out_TS_gamba)