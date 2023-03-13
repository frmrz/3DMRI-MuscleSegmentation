# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:25:46 2023

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

directory_COSCIA = "C:/Users/omobo/Desktop/PROVA HM/histogram_matched/COSCIA/FSHD"
#ref_path = "C:/Users/omobo/Desktop/PROVA HM/histogram_matched/COSCIA/FSHD/1_022_1_1_L.nii"
directory_GAMBA = "C:/Users/omobo/Desktop/PROVA HM/histogram_matched/GAMBA/FSHD"


"""PRECEDENTEMENTE HO FATTO IL RESIZE DEI VOLUMI 160X200X72 PER OTTENERE -> 256X192X32 PER FARE L'HM. 
POI HO RIPRISTINATO LE DIMENSIONI ORIGINALI E HO CALCOLATO IL NUOVO ISTOGRAMMA DI RIFERIMENTO
HO ESEGUITO L'HM TRA IL NUOVO RIFERIMENTO E IL VOLUME 160X200X72 ORIGINALE, SENZA PORTARE AVANTI LE 
MODIFICHE DOVUTE AL RESIZE."""



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

#ref_COSCIA = average_ref(directory_COSCIA)

ref_GAMBA = average_ref(directory_GAMBA)

# ref_COSCIA =  sitk.ReadImage(ref_path,sitk.sitkFloat32)
# ref_COSCIA = sitk.GetArrayFromImage(ref_COSCIA)
#%% definisco le cartelle utili
input_dir = "C:/Users/omobo/Desktop/PROVA HM"

#in_FSHD_coscia = os.path.join(input_dir,'FSHD (160,200,72)/COSCIA')
# in_MD1_coscia = os.path.join(input_dir,'MD1/COSCIA')
# in_HV_coscia = os.path.join(input_dir,'HV/COSCIA')

in_FSHD_gamba = os.path.join(input_dir,'FSHD (160,200,72)/GAMBA')
# in_MD1_gamba = os.path.join(input_dir,'MD1/GAMBA')
# in_HV_gamba = os.path.join(input_dir,'HV/GAMBA') 

output_dir = "C:/Users/omobo/Desktop/PROVA HM/histogram_matched"

#out_FSHD_coscia = os.path.join(output_dir,'COSCIA/FSHD_ULTIMO')
# out_MD1_coscia = os.path.join(output_dir,'COSCIA/MD1')
# out_HV_coscia = os.path.join(output_dir,'COSCIA/HV')

out_FSHD_gamba = os.path.join(output_dir,'GAMBA/FSHD_ULTIMO')
# out_MD1_gamba = os.path.join(output_dir,'GAMBA/MD1')
# out_HV_gamba = os.path.join(output_dir,'GAMBA/HV') 

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
#save_histogram_match(in_FSHD_coscia,ref_COSCIA,out_FSHD_coscia)
# save_histogram_match(in_MD1_coscia,ref_COSCIA,out_MD1_coscia)
# save_histogram_match(in_HV_coscia,ref_COSCIA,out_HV_coscia)

# # #GAMBA
save_histogram_match(in_FSHD_gamba,ref_GAMBA,out_FSHD_gamba)
# save_histogram_match(in_MD1_gamba,ref_GAMBA,out_MD1_gamba)
# save_histogram_match(in_HV_gamba,ref_GAMBA,out_HV_gamba)


#%%RIPRISTINARE LE CORRETTE DIMENSIONI DEI VOLUMI FSHD DOPO L'HM

    #PRENDO I VOLUMI DA out_FSHD_coscia e out_FSHD_gamba e ne ripristino le dimensioni
    #da 256,192,32 per eseguire l'hm su tutti i volumi, a 160,200,72 per non perdere troppe informazioni spaziali
    #e iutilizzare le vecchie label
    
    
# def resize_FSHD(input_dir,output_dir,spatial_size):
#     IMG_path = sorted(glob(os.path.join(input_dir,'*.nii')))
#     files = [{"image": image_name,"original":image_name} for image_name in IMG_path]

#     #FACCIO LE TRASFORMAZIONI SOLO SU IMAGE
#     transforms = Compose([LoadImaged(keys=['image','original']),
#                           EnsureChannelFirstd(keys=["image","original"]),
#                           Resized(keys=["image"],spatial_size=spatial_size)])


#     ds = Dataset(files ,transforms)
#     loader = DataLoader(ds)

#     # train_patient = first(loader)

#     # print(train_patient["image"].shape,train_patient["original"].shape)

#     # slice_n = 15

#     # plt.figure('prova',(12,6))


#     # plt.subplot(1,2,1)
#     # plt.title('FILE TRASFORMATO')
#     # plt.imshow(train_patient["image"][0,0,:,:,slice_n],cmap='gray')

#     # plt.subplot(1,2,2)
#     # plt.title('ORIGINALE')
#     # plt.imshow(train_patient["original"][0,0,:,:,slice_n],cmap='gray')

#     # plt.show()


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#     for batch in loader:
#         image_resized = batch["image"].to(device)
#         #image_original = batch["original"].to(device)
#         img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
#         affine = batch['image_meta_dict']['affine'][0].numpy()  
        
#         image_resized_array = image_resized.cpu().numpy()[0,0,:,:,:]
       
#         final_image = nib.Nifti1Image(image_resized_array, affine)
#         nib.save(final_image,os.path.join(output_dir,'{}.nii'.format(img_name.split('.')[0])))
#         print('Image {}.nii Saved!'.format(img_name.split('.')[0]))
        
        
# #%%

# resize_FSHD(out_FSHD_coscia,out_FSHD_coscia,(160,200,72))
# resize_FSHD(out_FSHD_gamba,out_FSHD_gamba,(160,200,72))

#%%

input_dir = in_FSHD_gamba
mask_dir= "C:/Users/omobo/Desktop/PROVA HM/FSHD (160,200,72)/mask/gamba"
hm_dir = out_FSHD_gamba
 

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
    hm_dir = out_FSHD_gamba

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
plot_orig_vs_hm(input_dir,mask_dir)
