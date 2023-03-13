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
import nibabel as nib

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
    Resized,
    Spacingd

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

output_dir = 'C:/Users/omobo/Desktop/PROVA HM/FSHD (160,200,72)_resized(256,192,32)'
if not os.path.exists(output_dir):
   os.mkdir(output_dir)

output_dir_COSCIA = os.path.join(output_dir,'COSCIA')
if not os.path.exists(output_dir_COSCIA):
   os.mkdir(output_dir_COSCIA)
   
output_dir_GAMBA = os.path.join(output_dir,'GAMBA') 
if not os.path.exists(output_dir_GAMBA):
   os.mkdir(output_dir_GAMBA)


def resize_FSHD(input_dir,output_dir,spatial_size):
    IMG_path = sorted(glob(os.path.join(input_dir,'*.nii')))
    files = [{"image": image_name,"original":image_name} for image_name in IMG_path]

    #FACCIO LE TRASFORMAZIONI SOLO SU IMAGE
    transforms = Compose([LoadImaged(keys=['image','original']),
                          EnsureChannelFirstd(keys=["image","original"]),
                          Resized(keys=["image"],spatial_size=spatial_size)])


    ds = Dataset(files ,transforms)
    loader = DataLoader(ds)

    train_patient = first(loader)

    print(train_patient["image"].shape,train_patient["original"].shape)

    slice_n = 15

    plt.figure('prova',(12,6))


    plt.subplot(1,2,1)
    plt.title('DOPO HM')
    plt.imshow(train_patient["original"][0,0,:,:,slice_n],cmap='gray')
    
    plt.subplot(1,2,2)
    plt.title('DIMENSIONI RIPRISTINATE')
    plt.imshow(train_patient["image"][0,0,:,:,slice_n],cmap='gray')

    plt.show()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    for batch in loader:
        image_resized = batch["image"].to(device)
        #image_original = batch["original"].to(device)
        img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        affine = batch['image_meta_dict']['affine'][0].numpy()  
        
        image_resized_array = image_resized.cpu().numpy()[0,0,:,:,:]
       
        final_image = nib.Nifti1Image(image_resized_array, affine)
        #nib.save(final_image,os.path.join(output_dir,'{}.nii'.format(img_name.split('.')[0])))
        print('Image {}.nii Saved!'.format(img_name.split('.')[0]))
        
        
#%%

resize_FSHD(directory_FSHD_COSCIA,output_dir_COSCIA,(256,192,32))
resize_FSHD(directory_FSHD_GAMBA,output_dir_GAMBA,(256,192,32))