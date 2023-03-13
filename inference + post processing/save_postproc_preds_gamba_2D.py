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
from monai.data import  Dataset,  DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    AsDiscreted,
    Activationsd,  


)


import cv2


#%% caricare i SET con l'opportuno pre-processing


#directory= 'C:/Users/omobo/Desktop/PROVE 2D/DATASET 2D/all image (inference)/GAMBE'
#directory= 'C:/Users/omobo/Desktop/PROVE 2D/DATASET 2D/all image (inference) HM/GAMBE'
directory= "C:/Users/omobo/Desktop/all image, spacing mediano/inference (test completo)"




test_IMG_path = sorted(glob(os.path.join(directory,'imagesTs-gamba','*.png')))
test_MASK_path = sorted(glob(os.path.join(directory,'labelsTs-gamba','*.png')))

# train_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(train_IMG_path,train_MASK_path)]
# val_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(val_IMG_path,val_MASK_path)]
test_files = [{"image": image_name, "label": mask_name} for image_name,mask_name in zip(test_IMG_path,test_MASK_path)]

#%% pre-processing

#altre trasformazioni non sono necessarie, perchè le patches di input sono già:
    #croppate
    #ricampionate
    #orientate
    #normalizzate a livello dell'istogramma di luminosità
    
transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        
    ]
)


test_ds = Dataset(test_files ,transforms)
test_loader = DataLoader( test_ds, batch_size=1)


#%%============================================================================
#                                 MODELLO
#==============================================================================

#model_dir = 'C:/Users/omobo/Desktop/PROVE 2D/MODEL/GAMBE'
#model_dir = 'C:/Users/omobo/Desktop/histogram matched/MODELLI/GAMBA 2D/resnet34_0_8180'
#model_dir = 'C:/Users/omobo/Desktop/no_zeros/GAMBA 0.8068'
model_dir = "C:/Users/omobo/Desktop/NUOVI MODELLI/GAMBA 2D"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = smp.Unet('resnet50',
                     encoder_depth = 4,
                     encoder_weights= 'imagenet',
                     decoder_channels= ( 128 , 64 , 32 ,16),
                     in_channels = 1, classes=10).to(device)

model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model_Unet_2d.pth"),map_location=torch.device('cpu')))

model.eval()
model.to(device)

#%% cartelle di salvataggio dei risultati

"""
Definisco le cartelle in cui salverò le maschere dopo il post-processing e se non esistono le creo
"""
output_dir = model_dir

# output_train_dir = os.path.join(output_dir , 'outputTr_gamba (post-proc)')
# if not os.path.exists(output_train_dir):
#   os.mkdir(output_train_dir)
  
# output_val_dir = os.path.join(output_dir , 'outputVl_gamba (post-proc)')
# if not os.path.exists(output_val_dir):
#   os.mkdir(output_val_dir)
  
output_test_dir = os.path.join(output_dir , 'outputTs_gamba (post-proc)')
if not os.path.exists(output_test_dir):
  os.mkdir(output_test_dir)

"""
Definisco le cartelle in cui salverò di nuovo le maschere manuali e se non esistono le creo
"""
#durante il caricamento le maschere vengono ruotate
#gli output della rete manterranno il nuovo orientamento ma questo crea problemi 
#quando dobbiamo calcolare le metricheperchè le maschere non si sovrapporranno
#allora ricarico le maschere manuali che verranno ruotate come le immagini 
#e le risalvo così da mantenere il nuovo orientamento e calcolare correttamente le metriche

# label_train_dir = os.path.join(output_dir , 'labelTr_gamba')
# if not os.path.exists(label_train_dir):
#   os.mkdir(label_train_dir)
  
# label_val_dir = os.path.join(output_dir , 'labelVl_gamba')
# if not os.path.exists(label_val_dir):
#   os.mkdir(label_val_dir)
  
label_test_dir = os.path.join(output_dir , 'labelTs_gamba')
if not os.path.exists(label_test_dir):
  os.mkdir(label_test_dir)
  
#%%============================================================================
#                  INFERENCE, POST-PROCESSING E SALVATAGGIO
#==============================================================================


post_pred = Compose([
    
    Activationsd(keys="pred",softmax=True),
    AsDiscreted(keys="pred",argmax=True),

    ])

inferer = SimpleInferer()

def save_post_proc(loader, post_pred, output_dir,label_dir, device, model):
    print("------------------------------")
    with torch.no_grad():
        
            for batch in loader:  
                inputs = batch["image"].to(device)
                img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])   #-->2_023_1_2_47.nii
                image_id = os.path.splitext(img_name)[0]   #-->2_023_1_2_47
                patient = img_name.split('_')[1]  #--> 023
                
                mask_name = os.path.basename(batch["label_meta_dict"]["filename_or_obj"][0])
                label_ = batch["label"].detach().numpy()[0,0,:,:]
                label_ = label_.astype(np.uint8)
                print("Inference on patient {}".format(patient))
                #batch["pred"] = sliding_window_inference(inputs, (160,160), 4, model)
                batch["pred"] = inferer(inputs, model)
               
                batch = [post_pred(i) for i in decollate_batch(batch)] #batch adesso è una lista
                           
                outputs = (batch[0]["pred"].to(device)).cpu().numpy()[0]  #-->ottengo un array 2D
                
                print('Post-processing')
                
                data_array = outputs.astype(np.uint8)
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
               
                #----------------------------------------------------------------
                
                #salvo l'output post-processato nelle apposite cartelle di output
                cv2.imwrite(os.path.join(output_dir,'{}_pred.png'.format(image_id)),post_proc_slice )
                
                #risalvo le maschere manuali così hanno lo stesso orientamento di quelle automatiche
                cv2.imwrite(os.path.join(label_dir,mask_name),label_ )
               
                print('Image {}_pred.png Saved!'.format(image_id))
            


## TRAIN
# save_post_proc(train_loader, post_pred ,output_train_dir,label_train_dir, device, model)

# #VAL
# save_post_proc(val_loader, post_pred,output_val_dir,label_val_dir, device, model)

#TEST
save_post_proc(test_loader, post_pred,output_test_dir,label_test_dir, device, model)
