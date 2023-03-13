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

#Dopo aver eseguito il resize dei volumi FSHD (da 160,200,72 a 256,192,32)
#li ho raccolti tutti in una cartella (directory_COSCIA e directory_GAMBA)
#poi tutti questi volumi li ho raccolti, organizzati in batch e poi mediati per ottenere 
#l'istogramma di riferimento medio, per coscia e gamba.


directory_COSCIA = "C:/Users/omobo/Desktop/PROVA HM/TUTTE/COSCIA"
directory_GAMBA = "C:/Users/omobo/Desktop/PROVA HM/TUTTE/GAMBA"


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

#%% VOLUMI DI RIFERIMENTO PER GAMBA E COSCIA
#questi volumi di riferimento saranno utilizzati per fare istogram matching

ref_COSCIA = average_ref(directory_COSCIA)

ref_GAMBA = average_ref(directory_GAMBA)

#%% definisco le cartelle utili
input_dir = "C:/Users/omobo/Desktop/PROVA HM"

in_FSHD_coscia = os.path.join(input_dir,'FSHD/COSCIA')
in_MD1_coscia = os.path.join(input_dir,'MD1/COSCIA')
in_HV_coscia = os.path.join(input_dir,'HV/COSCIA')

in_FSHD_gamba = os.path.join(input_dir,'FSHD/GAMBA')
in_MD1_gamba = os.path.join(input_dir,'MD1/GAMBA')
in_HV_gamba = os.path.join(input_dir,'HV/GAMBA') 

output_dir = "C:/Users/omobo/Desktop/PROVA HM/histogram_matched"

out_FSHD_coscia = os.path.join(output_dir,'COSCIA/FSHD')
out_MD1_coscia = os.path.join(output_dir,'COSCIA/MD1')
out_HV_coscia = os.path.join(output_dir,'COSCIA/HV')

out_FSHD_gamba = os.path.join(output_dir,'GAMBA/FSHD')
out_MD1_gamba = os.path.join(output_dir,'GAMBA/MD1')
out_HV_gamba = os.path.join(output_dir,'GAMBA/HV') 

#%%


def save_histogram_match(img_dir,ref,output_path):
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
        
        ref_array=ref
            
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
        

        
#%% APPLICO LA FUNZIONE

#save_histogram_match(img_dir,ref_FSHD,ref,output_path)

#COSCIA
save_histogram_match(in_FSHD_coscia,ref_COSCIA,out_FSHD_coscia)
save_histogram_match(in_MD1_coscia,ref_COSCIA,out_MD1_coscia)
save_histogram_match(in_HV_coscia,ref_COSCIA,out_HV_coscia)

# #GAMBA
save_histogram_match(in_FSHD_gamba,ref_GAMBA,out_FSHD_gamba)
save_histogram_match(in_MD1_gamba,ref_GAMBA,out_MD1_gamba)
save_histogram_match(in_HV_gamba,ref_GAMBA,out_HV_gamba)


#%%RIPRISTINARE LE CORRETTE DIMENSIONI DEI VOLUMI FSHD DOPO L'HM

    #PRENDO I VOLUMI DA out_FSHD_coscia e out_FSHD_gamba e ne ripristino le dimensioni
    #da 256,192,32 per eseguire l'hm su tutti i volumi, a 160,200,72 per non perdere troppe informazioni spaziali
    #e iutilizzare le vecchie label
    
    
def resize_FSHD(input_dir,output_dir,spatial_size):
    IMG_path = sorted(glob(os.path.join(input_dir,'*.nii')))
    files = [{"image": image_name,"original":image_name} for image_name in IMG_path]

    #FACCIO LE TRASFORMAZIONI SOLO SU IMAGE
    transforms = Compose([LoadImaged(keys=['image','original']),
                          EnsureChannelFirstd(keys=["image","original"]),
                          Resized(keys=["image"],spatial_size=spatial_size,mode = 'bilinear')])


    ds = Dataset(files ,transforms)
    loader = DataLoader(ds)

    # train_patient = first(loader)

    # print(train_patient["image"].shape,train_patient["original"].shape)

    # slice_n = 15

    # plt.figure('prova',(12,6))


    # plt.subplot(1,2,1)
    # plt.title('FILE TRASFORMATO')
    # plt.imshow(train_patient["image"][0,0,:,:,slice_n],cmap='gray')

    # plt.subplot(1,2,2)
    # plt.title('ORIGINALE')
    # plt.imshow(train_patient["original"][0,0,:,:,slice_n],cmap='gray')

    # plt.show()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for batch in loader:
        image_resized = batch["image"].to(device)
        #image_original = batch["original"].to(device)
        img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        affine = batch['image_meta_dict']['affine'][0].numpy()  
        
        image_resized_array = image_resized.cpu().numpy()[0,0,:,:,:]
       
        final_image = nib.Nifti1Image(image_resized_array, affine)
        nib.save(final_image,os.path.join(output_dir,'{}.nii'.format(img_name.split('.')[0])))
        print('Image {}.nii Saved!'.format(img_name.split('.')[0]))
        
        
#%%

resize_FSHD(out_FSHD_coscia,out_FSHD_coscia,(160,200,72))
resize_FSHD(out_FSHD_gamba,out_FSHD_gamba,(160,200,72))
    
