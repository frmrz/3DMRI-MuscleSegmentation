# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:45:19 2022

@author: omobo
"""


import numpy as np
import os
from glob import glob
from skimage.morphology import remove_small_objects,label, area_closing
from skimage import segmentation
import segmentation_models_pytorch as smp

from monai.data import  decollate_batch

import torch

from monai.inferers import sliding_window_inference, SimpleInferer
from monai.data import  Dataset,  DataLoader, PILReader
from monai.transforms import (
    Compose,
    
    LoadImaged,
    EnsureChannelFirstd,
    AsDiscrete,
    Activations,
    EnsureType,
    ToTensord


)
from monai.utils import  set_determinism
import SimpleITK as sitk

import cv2
import matplotlib.pyplot as plt
from skimage.transform import rotate

import PIL

#%% caricare i SET con l'opportuno pre-processing

inferer = SimpleInferer()
SEED = 3 
set_determinism(SEED)


#directory= 'C:/Users/omobo/Desktop/PROVE 2D/DATASET 2D/all image (inference)/COSCE'
directory= 'C:/Users/omobo/Desktop/PROVE 2D/DATASET 2D/all image (inference) HM/COSCE'
#directory= "C:/Users/omobo/Desktop/all image, spacing mediano/inference (test completo)"


test_IMG_path = sorted(glob(os.path.join(directory,'imagesTs-coscia','*.png')))
test_MASK_path = sorted(glob(os.path.join(directory,'labelsTs-coscia','*.png')))


test_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path,test_MASK_path)]

#%% pre-processing

#altre trasformazioni non sono necessarie, perchè le patches di input sono già:
    #croppate
    #ricampionate
    #orientate
    #normalizzate a livello dell'istogramma di luminosità 
    
transforms = Compose(
    [
        LoadImaged(keys=["image", "label"],image_only=False,reader=PILReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        #ToTensord(keys=["image", "label"])
    ]
)



test_ds = Dataset(test_files ,transforms)
test_loader = DataLoader( test_ds, batch_size=1)


#%%============================================================================
#                             IMPORT MODELLO
#==============================================================================


#model_dir = 'C:/Users/omobo/Desktop/PROVE 2D/MODEL/COSCIA'
#model_dir = 'C:/Users/omobo/Desktop/histogram matched/MODELLI/COSCIA 2D/coscia_resnet34_0_8248'
model_dir = 'C:/Users/omobo/Desktop/no_zeros/COSCIA 0.8339 e 0.8347'
#model_dir = "C:/Users/omobo/Desktop/NUOVI MODELLI/COSCIA 2D"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = smp.Unet('resnet50',
                     encoder_depth = 4,
                     encoder_weights= 'imagenet',
                     decoder_channels= ( 128 , 64 , 32 ,16),
                     in_channels = 1, classes=13).to(device)


model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model_Unet_2d.pth"),map_location=torch.device('cpu')))

model.eval()
model.to(device)

#%% cartelle di salvataggio dei risultati

"""
Definisco le cartelle in cui salverò le maschere dopo il post-processing e se non esistono le creo
"""
output_dir = model_dir

# output_train_dir = os.path.join(output_dir , 'outputTr_coscia (post-proc)')
# if not os.path.exists(output_train_dir):
#   os.mkdir(output_train_dir)
  
# output_val_dir = os.path.join(output_dir , 'outputVl_coscia (post-proc)')
# if not os.path.exists(output_val_dir):
#   os.mkdir(output_val_dir)
  
output_test_dir = os.path.join(output_dir , 'outputTs_coscia (post-proc)')
if not os.path.exists(output_test_dir):
  os.mkdir(output_test_dir)

#durante il caricamento le maschere vengono ruotate
#gli output della rete manterranno il nuovo orientamento ma questo crea problemi 
#quando dobbiamo calcolare le metriche perchè le maschere non si sovrapporranno
#allora ricarico le maschere manuali che verranno ruotate come le immagini 
#e le risalvo così da mantenere il nuovo orientamento e calcolare correttamente le metriche

# label_train_dir = os.path.join(output_dir , 'labelTr_coscia')
# if not os.path.exists(label_train_dir):
#   os.mkdir(label_train_dir)
  
# label_val_dir = os.path.join(output_dir , 'labelVl_coscia')
# if not os.path.exists(label_val_dir):
#   os.mkdir(label_val_dir)
  
label_test_dir = os.path.join(output_dir , 'labelTs_coscia')
if not os.path.exists(label_test_dir):
  os.mkdir(label_test_dir)
  
#%%============================================================================
#                  INFERENCE, POST-PROCESSING E SALVATAGGIO
#==============================================================================

output_dir_aiuto = "C:/Users/omobo/Desktop/no_zeros/COSCIA 0.8339 e 0.8347/aiuto"
post_pred = Compose([
    
    EnsureType(),
    Activations(softmax=True),
    AsDiscrete(argmax=True),

    ])

inferer = SimpleInferer()

#     DEBUG

for batch in test_loader:  
    inputs,labels = (batch["image"].to(device),batch["label"].to(device))
    img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])   #-->2_023_1_2_47.nii
    
#------------------------------------------------------------------------------    
    
    image_id = os.path.splitext(img_name)[0]   #-->2_023_1_2_47
    patient = img_name.split('_')[1]  #--> 023
    print(image_id)
       
    print("Inference on patient {}".format(patient))
    outputs = inferer(inputs, model)
    
    
    
    # a = outputs.detach().numpy()
    # b = batch["label"].detach().numpy()
    #batch['pred'] = torch.softmax(batch['pred'],dim=1)
    
    
    
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 2, 2)
    # plt.title("{}_mask".format(image_id))
    # plt.imshow(b[0,0,:,:])
    # plt.show()
    
    outputs = [post_pred(i) for i in decollate_batch(outputs)] 
    
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 2, 2)
    # plt.title("{}_mask".format(image_id))
    # plt.imshow(outputs[0][0])
    # plt.show()
               
    outputs = outputs[0][0].cpu().numpy().squeeze()
    
    #
    
    # print('Post-processing')
    
    # data_array = outputs
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(data_array)
    # plt.show()
    # #------------------------------------------------------------------                    
    # # remove small objects
    # array = label(data_array)
    # obj_removed = remove_small_objects (array , min_size = 60 ) 
    # object_removed = segmentation.watershed(
    #           obj_removed, data_array, mask= obj_removed)
    
    # # fill holes
    # post_proc_slice = area_closing(object_removed, area_threshold = 40)
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(post_proc_slice)
    # plt.show()
    # #------------------------------------------------------------------
    # post_proc_slice = post_proc_slice*255
    #salvo l'output post-processato nelle apposite cartelle di output
    
    post_proc_slice = rotate(outputs,90).astype(np.uint8)   #monai ruota le patch quando le carica. ripristino l'orientamento
    post_proc_slice = np.flipud(post_proc_slice)  #e poi le flippo
    
    stackPIL = PIL.Image.fromarray(post_proc_slice)
    stackPIL.save(os.path.join(output_dir_aiuto,'{}.png'.format(image_id)))
    
    print('Image {}.png Saved!'.format(image_id))

