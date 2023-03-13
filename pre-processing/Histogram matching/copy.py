# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:22:03 2023

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
import shutil

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

output_dir = "C:/Users/omobo/Desktop/nuovo_hm"

out_TR_coscia = os.path.join(output_dir,'imagesTr-coscia')
out_VL_coscia = os.path.join(output_dir,'imagesVl-coscia')



def copy(input_dir,output_dir):
    img_dir = input_dir
    volums = os.listdir(img_dir)

    hm_dir = "C:/Users/omobo/Desktop/PROVA HM/TUTTE_HM"

    for img in volums:
        
        print(img)
        
        from_path = os.path.join(hm_dir,img)  #percorso del nuovo file hm
        to_path = output_dir
        shutil.copy2(from_path,to_path)


copy(in_TR_coscia,out_TR_coscia)
copy(in_VL_coscia,out_VL_coscia)    
    