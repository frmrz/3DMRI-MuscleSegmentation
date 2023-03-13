# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:17:45 2022

@author: Evelin
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from skimage.morphology import remove_small_objects,label, area_closing
from skimage import segmentation
from glob import glob

from monai.data import  decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import torch
import pandas as pd

from monai.inferers import sliding_window_inference
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
    SaveImaged,
    Activationsd,  

)
from monai.utils import  set_determinism
from monai.networks.nets import SwinUNETR,UNet
from monai.networks.layers import Norm

from functions_2 import save_post_proc, make_dict, calc_metrics, mean_muscle_metrics,std_muscle_metrics, plot_results

#%% 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


#DECOMMENTARE PER INFERENCE POST-PROCESSING E SALVATAGGIO
# #----------------------------------------------------------------------------------------------------------------------------
# SEED = 3
# set_determinism(SEED)


# directory_mask = 'C:/Users/omobo/Desktop/DATASET 3D (flip) R'
# directory = 'C:/Users/omobo/Desktop/histogram matched'

# train_IMG_path = sorted(glob(os.path.join(directory,'imagesTr-coscia','*.nii')))
# train_MASK_path = sorted(glob(os.path.join(directory_mask ,'labelsTr-coscia','*.nii')))


# val_IMG_path = sorted(glob(os.path.join(directory,'imagesVl-coscia','*.nii')))
# val_MASK_path = sorted(glob(os.path.join(directory_mask ,'labelsVl-coscia','*.nii')))


# test_IMG_path = sorted(glob(os.path.join(directory,'imagesTs-coscia','*.nii')))
# test_MASK_path = sorted(glob(os.path.join(directory_mask,'labelsTs-coscia','*.nii')))

# train_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path,train_MASK_path)]
# val_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path,val_MASK_path)]
# test_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path,test_MASK_path)]

# #%% pre-processing
# transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         CropForegroundd(keys=["image"], source_key="image"),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         Spacingd(keys=["image"], pixdim=(
#             1.0938,1.0938,5.0), mode=("bilinear")),
#     ]
# )


# #applicare il pre-processing

# train_ds = Dataset(train_files ,transforms)
# train_loader = DataLoader( train_ds, batch_size=1)

# val_ds = Dataset(val_files ,transforms)
# val_loader = DataLoader( val_ds, batch_size=1)

# test_ds = Dataset(test_files ,transforms)
# test_loader = DataLoader( test_ds, batch_size=1)


# #%% caricamento del modello

# """
# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
# Il modello è stato salvato come state_dict. quindi non possiamo caricarlo direttamente 
# ma prima dobbiamo definire la backbone (una SwinUNETR) su cui andremo a caricare i pesi
 
# """

# #model_dir#
# #model_dir = 'C:/Users/omobo/Desktop/histogram matched/MODELLI/COSCIA 3D/SwinUNETR 0.8213'
# #model_dir = 'C:/Users/omobo/Desktop/histogram matched/MODELLI/COSCIA 3D/UNet 0.8073'
# #model_dir = "C:/Users/omobo/Desktop/NUOVI MODELLI/COSCIA 3D/UNet spacing 1.0938"
# #model_dir = "C:/Users/omobo/Desktop/NUOVI MODELLI/COSCIA 3D/SwinUNETR spacing 1.0938"
# model_dir = "C:/Users/omobo/Desktop/nuovo_hm/modelli"

# # model = SwinUNETR(
# #     img_size=(96, 96, 32),
# #     in_channels=1,
# #     out_channels=13,
# #     feature_size=24,  #32 --> ValueError: feature_size should be divisible by 12.
# #     use_checkpoint=True,
# # ).to(device)

# model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=13,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm=Norm.BATCH,
# ).to(device)
# model.load_state_dict(torch.load(
#     os.path.join(model_dir, "best_metric_model_UNET_coscia_13594.pth"),map_location=torch.device('cpu')))


# model.eval()
# model.to(device)

# #%% cartelle di salvataggio dei risultati

# """
# Definisco le cartelle in cui salverò le maschere dopo il post-processing e se non esistono le creo
# """

# # output_train_dir = os.path.join(model_dir , 'outputTr_coscia (post-proc)')
# # if not os.path.exists(output_train_dir):
# #   os.mkdir(output_train_dir)
  
# # output_val_dir = os.path.join(model_dir , 'outputVl_coscia (post-proc)')
# # if not os.path.exists(output_val_dir):
# #   os.mkdir(output_val_dir)
  
# # output_test_dir = os.path.join(model_dir , 'outputTs_coscia (post-proc)')
# # if not os.path.exists(output_test_dir):
# #   os.mkdir(output_test_dir)

# output_test_dir = "C:/Users/omobo/Desktop/VOLUMI/COSCIA/output_unet_3D"
# #output_test_dir = "C:/Users/omobo/Desktop/VOLUMI/COSCIA/output_swin_3D"
# #%%============================================================================
# #                  INFERENCE, POST-PROCESSING E SALVATAGGIO
# #==============================================================================


# post_pred = Compose([
#     Invertd(
#         keys="pred",
#         transform=transforms,
#         orig_keys="image",
#         meta_keys="pred_meta_dict",
#         orig_meta_keys="image_meta_dict",
#         meta_key_postfix="meta_dict",
#         nearest_interp=False,
#         to_tensor=True,
#     ),
#     Activationsd(keys="pred",softmax=True),
#     AsDiscreted(keys="pred",argmax=True)
#     ])


# #TRAIN
# # save_post_proc(train_loader, post_pred ,output_train_dir, device, model)

# # #VAL
# # save_post_proc(val_loader, post_pred,output_val_dir, device, model)

# #TEST
# save_post_proc(test_loader, post_pred,output_test_dir, device, model)
# #---------------------------------------------------------------------------------------------------------------------------------

#%%============================================================================
#                               CALCOLO METRICHE
#==============================================================================
#                          CAMBIARE LE CARTELLE QUI
#==============================================================================
#definisco i percorsi delle maschere post-processate salvate prima
# train_PRED_path = sorted(glob(os.path.join(output_train_dir,'*.nii')))

# val_PRED_path = sorted(glob(os.path.join(output_val_dir,'*.nii')))

#test_PRED_path = sorted(glob(os.path.join(output_test_dir,'*.nii')))

#output_dir = "C:/Users/omobo/Desktop/VOLUMI/COSCIA/vol_cropped"


# out_label = os.path.join(output_dir,'labels')
# out_pred = os.path.join(output_dir,'mask_resnet')
#out_pred = os.path.join(output_dir,'mask_swin')

#out_label = os.path.join(output_dir,'labels_3D')
out_label = "C:/Users/omobo/Desktop/FILL_CENTROIDI/COSCIA/vol_cropped/label"
#out_pred = os.path.join(output_dir,'output_unet_3D')
#out_pred = os.path.join(output_dir,'output_swin_3D')
out_pred = "C:/Users/omobo/Desktop/FILL_CENTROIDI/COSCIA/vol_cropped/SWIN"

directory = "C:/Users/omobo/Desktop/Unet-resnet/COSCIA/vol_cropped"
# # #directory = "C:/Users/omobo/Desktop/Unet-resnet/GAMBA/vol_cropped"


out_label = os.path.join(directory,'label_vol')
out_pred = os.path.join(directory,'pred_vol')


test_MASK_path = sorted(glob(os.path.join(out_label,'*.nii')))
test_PRED_path = sorted(glob(os.path.join(out_pred,'*.nii')))

# train_files = [{"pred": pred_name, "label": mask_name} for pred_name,mask_name in zip(train_PRED_path,train_MASK_path)]
# val_files = [{"pred": pred_name, "label": mask_name} for pred_name,mask_name in zip(val_PRED_path,val_MASK_path)]
test_files = [{"pred": pred_name, "label": mask_name} for pred_name,mask_name in zip(test_PRED_path,test_MASK_path)]

#%%
"""
 CARICO:
     LE MASCHERE POST-PROCESSATE--> "pred"
     LE SEGMENTAZIONI MANUALI--> "label"
"""
print("Metrics")

transforms = Compose(
    [
        LoadImaged(keys=["pred", "label"]),
        EnsureChannelFirstd(keys=["pred", "label"]),
       
    ])


# train_ds = Dataset(train_files ,transforms)
# train_loader = DataLoader( train_ds, batch_size=1)

# val_ds = Dataset(val_files ,transforms)
# val_loader = DataLoader( val_ds, batch_size=1)

test_ds = Dataset(test_files ,transforms)
test_loader = DataLoader( test_ds, batch_size=1)

#%%

"""
DiceMetric richiede che:
    -y_pred (maschera automatica) is expected to have binarized predictions
    -y (maschera manuale) should be in one-hot format 
"""

 
post_transform = Compose([AsDiscrete(to_onehot=13)])

thigh_muscles= ["Vasto Laterale",
    "Vasto Intermedio",
    "Vasto Mediale",
    "Retto femorale",
    "Muscolo sartorio",
    "Muscolo gracile",
    "Muscolo semimembranoso",
    "Muscolo semitendinoso",
    "Bicipite femorale - capo lungo",
    "Bicipite femorale - capo breve",
    "Grande Adduttore + adduttore corto",
    "Adduttore Lungo"]

#%% Creare i dictionaries delle metriche per TRAIN VAL E TEST
print("Make dictionaries")

#TRAIN
# TRAIN = {}
# TRAIN = make_dict(train_loader, TRAIN) 


# #VAL
# VAL = {}
# VAL = make_dict(val_loader, VAL)


#TEST
TEST = {}
TEST = make_dict(test_loader, TEST)


#%% Calcolo delle metriche divise per subset

#TRAIN , VAL , e TEST sono i dictionaries che contengono le metriche
#di ogni label per ogni soggetto

# TRAIN , dice_vals_batch_TR , hd95_vals_batch_TR , dice_vals_TR , hd95_vals_TR = calc_metrics(train_loader,
#                                                                                                device,
#                                                                                               post_transform, 
#                                                                                               TRAIN, thigh_muscles)


# VAL , dice_vals_batch_VL , hd95_vals_batch_VL , dice_vals_VL , hd95_vals_VL = calc_metrics(val_loader,
#                                                                                             device,
#                                                                                            post_transform, 
#                                                                                             VAL, thigh_muscles)



TEST , dice_vals_batch_TS , hd95_vals_batch_TS , dice_vals_TS , hd95_vals_TS = calc_metrics(test_loader,
                                                                                             device,
                                                                                            post_transform,
                                                                                            TEST, thigh_muscles)

#%% calcolo delle metriche di ogni label per tutti i soggetti
print("Mean muscle metrics")

# muscle_metric_TRAIN = {}
# muscle_metric_TRAIN = mean_muscle_metrics(dice_vals_batch_TR,
#                                           hd95_vals_batch_TR,
#                                           muscle_metric_TRAIN,
#                                           thigh_muscles)

# muscle_metric_VAL = {}
# muscle_metric_VAL = mean_muscle_metrics(dice_vals_batch_VL,
#                                         hd95_vals_batch_VL,
#                                         muscle_metric_VAL, 
#                                         thigh_muscles)

muscle_metric_TEST = {}
muscle_metric_TEST = mean_muscle_metrics(dice_vals_batch_TS,
                                         hd95_vals_batch_TS, 
                                         muscle_metric_TEST,
                                         thigh_muscles)

#%%

print("Standard dev muscle metrics")

# muscle_std_TRAIN = {}
# muscle_std_TRAIN = std_muscle_metrics(dice_vals_batch_TR,
#                                           hd95_vals_batch_TR,
#                                           muscle_std_TRAIN,
#                                           thigh_muscles)

# muscle_std_VAL = {}
# muscle_std_VAL = std_muscle_metrics(dice_vals_batch_VL,
#                                         hd95_vals_batch_VL,
#                                         muscle_std_VAL, 
#                                         thigh_muscles)

muscle_std_TEST = {}
muscle_std_TEST = std_muscle_metrics(dice_vals_batch_TS,
                                         hd95_vals_batch_TS, 
                                         muscle_std_TEST,
                                         thigh_muscles)

#%% calcolo delle metriche su tutte le label su tutti i soggetti (per ogni set)

# mean_dice_train = np.mean(dice_vals_TR)
# mean_hd95_train = np.mean(hd95_vals_TR)

# mean_dice_val = np.mean(dice_vals_VL)
# mean_hd95_val = np.mean(hd95_vals_VL)

# mean_dice_test = np.mean(dice_vals_TS)
# mean_hd95_test = np.mean(hd95_vals_TS)

# print(
#    'Overall Mean Dice:\nTRAIN = ', mean_dice_train,'\n VAL = ',mean_dice_val,'\nTEST = ',mean_dice_test
#       )
# print(
#    'Overall HD 95%: \nTRAIN = ', mean_hd95_train,'\n VAL = ',mean_hd95_val,'\nTEST = ',mean_hd95_test
#    )
    
#%%============================================================================
#                            VISUALIZZAZIONE METRICHE
#==============================================================================


labels = ["VL",                          # "Vasto Laterale",
          "VI",                          # "Vasto Intermedio",
          "VM",                          # "Vasto Mediale",
          "RF",                          # "Retto femorale",
          "SAR",                         # "Muscolo sartorio",
          "GRA",                         # "Muscolo gracile",
          "SEMI-MB",                     # "Muscolo semimembranoso",
          "SEMI-TN",                     # "Muscolo semitendinoso",
          "BF-CL",                       # "Bicipite femorale - capo lungo",
          "BF-CB",                       # "Bicipite femorale - capo breve",
          "ADD",                         # "Grande Adduttore + adduttore corto",
          "AL"]                          # "Adduttore Lungo"

legend = ["VL = Vasto Laterale", "VI = Vasto Intermedio","VM = Vasto Mediale",
          "RF = Retto femorale","SAR = Muscolo sartorio","GRA = Muscolo gracile", 
          "SEMI-MB = Muscolo semimembranoso","SEMI-TN = Muscolo semitendinoso",
          "BF-CL = Bicipite femorale - capo lungo",
          "BF-CB = Bicipite femorale - capo breve",
          "ADD = Grande Adduttore + adduttore corto",
          "AL = Adduttore Lungo"]

print("LEGEND")
for i in legend:
  print(i)
  
print("LABEL")
for i in labels:
  print(i)

# DICE
plot_results(labels,thigh_muscles,
             muscle_metric_TEST,
             muscle_std_TEST,
             'dice' , 'Dice')

# HD 95%
plot_results(labels,thigh_muscles,
            muscle_metric_TEST,
            muscle_std_TEST,
             'hd95' , 'HD 95%')

