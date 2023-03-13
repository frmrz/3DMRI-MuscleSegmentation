# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:46:28 2023

@author: omobo
"""

"""
ESTRAGGO LE SLICE DAI VOLUMI PER CONFRONTARLE CON QUELLE UTILIZZATE PER L'ALLENAMENTO 2D
"""
import numpy as np
import os
from glob import glob
import cv2


from monai.data import  Dataset,  DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    SpatialPadd,
    Spacingd
    

)

#from functions import  make_dict, calc_metrics, mean_muscle_metrics, plot_results

from skimage.morphology import area_closing,disk,binary_dilation
from skimage.measure import regionprops

from monai.utils import first
import shutil

#%% 
"""
Definisco le cartelle utili"""

directory = 'C:/Users/omobo/Desktop/histogram matched'
#directory_= 'C:/Users/omobo/Desktop/histogram matched/MODELLI/COSCIA 3D/SwinUNETR 0.8213'
#directory_= 'C:/Users/omobo/Desktop/histogram matched/MODELLI/COSCIA 3D/UNet 0.8073'

#directory_= 'C:/Users/omobo/Desktop/histogram matched/MODELLI/GAMBA 3D/SwinUNETR 0.7801'
directory_= 'C:/Users/omobo/Desktop/histogram matched/MODELLI/GAMBA 3D/UNet 0.7651'

directory_mask= 'C:/Users/omobo/Desktop/DATASET 3D (flip) R'


patches_dir = 'C:/Users/omobo/Desktop/all image'
if not os.path.exists(patches_dir):
  os.mkdir(patches_dir)
  
#%% creazione cartelle

patches_dir_coscia = os.path.join(patches_dir,'COSCE')
patches_dir_gamba = os.path.join(patches_dir,'GAMBE')

if not os.path.exists(patches_dir_coscia):
  os.mkdir(patches_dir_coscia)
if not os.path.exists(patches_dir_gamba):
  os.mkdir(patches_dir_gamba)


if not os.path.isdir(os.path.join(patches_dir_coscia,'imagesTs-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'imagesTs-coscia'))

if not os.path.isdir(os.path.join(patches_dir_gamba,'imagesTs-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'imagesTs-gamba'))   


if not os.path.isdir(os.path.join(patches_dir_coscia,'outputsTs-coscia')):
      os.mkdir(os.path.join(patches_dir_coscia,'outputsTs-coscia'))
     

if not os.path.isdir(os.path.join(patches_dir_gamba,'outputsTs-gamba')):
      os.mkdir(os.path.join(patches_dir_gamba,'outputsTs-gamba'))  


img_coscia_TS = 'C:/Users/omobo/Desktop/all image/COSCE/imagesTs-coscia'
mask_coscia_TS ='C:/Users/omobo/Desktop/all image/COSCE/outputsTs-coscia'


img_gamba_TS = 'C:/Users/omobo/Desktop/all image/GAMBE/imagesTs-gamba'
mask_gamba_TS ='C:/Users/omobo/Desktop/all image/GAMBE/outputsTs-gamba'


#%% 


test_IMG_path_g = sorted(glob(os.path.join(directory,'imagesTs-gamba','*.nii')))
test_MASK_path_g = sorted(glob(os.path.join(directory_mask,'labelsTs-gamba','*.nii')))
test_PRED_path_g = sorted(glob(os.path.join(directory_,'outputTs_gamba (post-proc)','*.nii')))

test_IMG_path_c = sorted(glob(os.path.join(directory,'imagesTs-coscia','*.nii')))
test_MASK_path_c = sorted(glob(os.path.join(directory_mask,'labelsTs-coscia','*.nii')))
test_PRED_path_c = sorted(glob(os.path.join(directory_,'outputTs_coscia (post-proc)','*.nii')))

test_files_c = [{"image": image_name, "label": mask_name, "pred": pred_name} for image_name,mask_name,pred_name in zip(test_IMG_path_c,test_MASK_path_c, test_PRED_path_c)]
test_files_g = [{"image": image_name, "label": mask_name, "pred": pred_name} for image_name,mask_name,pred_name in zip(test_IMG_path_g,test_MASK_path_g, test_PRED_path_g)]


#%% 


transforms_c = Compose(
    [
        LoadImaged(keys=["image", "label","pred"]),
        EnsureChannelFirstd(keys=["image", "label","pred"]),
        Spacingd(keys=["image", "label","pred"], pixdim=(
            1.0938,1.0938,5.0), mode=("bilinear","nearest","nearest")),
        SpatialPadd(keys=["image", "label","pred"],spatial_size = (300,300,-1))
        
    ]
)


transforms_g = Compose(
    [
        LoadImaged(keys=["image", "label","pred"]),
        EnsureChannelFirstd(keys=["image", "label","pred"]),
        Spacingd(keys=["image", "label","pred"], pixdim=(
            1.0,1.0,5.0), mode=("bilinear","nearest","nearest")),
        SpatialPadd(keys=["image", "label","pred"],spatial_size = (300,300,-1))
        
    ]
)


#COSCIA   

   
test_ds_c = Dataset(test_files_c , transforms_c)
test_loader_c = DataLoader( test_ds_c, batch_size=1)
   
#------------------------------------------------------------------------------
#GAMBA

   
test_ds_g = Dataset(test_files_g , transforms_g)
test_loader_g = DataLoader( test_ds_g, batch_size=1)
#------------------------------------------------------------------------------

footprint = disk(5)


first = first(test_loader_c)
#%% FUNZIONI UTILI

def centered_crop(img_array,mask_array,pred_array,width,height,x_c,y_c):
    
    
    binary_mask = np.copy(mask_array)
    
    binary_mask[binary_mask>0]=1
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(binary_mask, cmap="gray")
    # plt.show()
    filled_mask = binary_dilation(binary_mask,footprint=footprint)
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(filled_mask, cmap="gray")
    # plt.show()
    filled_mask = area_closing(filled_mask, area_threshold = 700)
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(filled_mask, cmap="gray")
    # plt.show()
    filled_mask = filled_mask.astype(np.uint8)
    properties = regionprops(filled_mask)
    if sum(sum(mask_array)) > 5000:
        y,x = properties[0].centroid  #se la maschera non è vuotaricalcola i centroidi della nuova maschera, altrimenti usa quelli di prima
        # fig, ax = plt.subplots()
        # ax.imshow(filled_mask, cmap=plt.cm.gray)
        # ax.plot(x_c, y_c, '.r', markersize=15)
        # plt.show
        x_c.append(x)
        y_c.append(y)
    
    width = width
    height = height
    
    ymin = int(y_c[-1] - width/2) 
    ymax = int(y_c[-1] + width/2) 
    xmin = int(x_c[-1] - width/2) 
    xmax = int(x_c[-1] + width/2) 
    
    img_patch = img_array[ymin:ymax,xmin:xmax]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(img_patch, cmap="gray")
    # plt.show()
    mask_patch = pred_array[ymin:ymax,xmin:xmax]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 1, 1)
    # plt.title("check image")
    # plt.imshow(mask_patch, cmap="gray")
    # plt.show()
    
    return img_patch,mask_patch



def percentile(image_array,mask_array):
    
    image_array_1D = image_array[mask_array != 0]   #individuo solo la regione dell'immagine che contiene i muscoli
    #calcolo il 1° e il 99° percentile di questa regione
    perc_1 = np.percentile(image_array_1D,1)
    perc_99 = np.percentile(image_array_1D,99)
    
    return perc_1,perc_99

#equivalent of imadjust in MATLAB
#https://stackoverflow.com/questions/49656244/fast-imadjust-in-opencv-and-python
def imadjust(slice_n,perc_1,perc_99):
    slice_adj = (slice_n - perc_1) * (255 / (perc_99 - perc_1))
    slice_adj = np.clip(slice_adj, 0, 255) # in-place clipping
    return slice_adj

    
def make_slices(loader,img_dir,mask_dir,width,height):
    for batch in loader:
        img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        image_id = os.path.splitext(img_name)[0]
        vol_array = batch['image'][0,0,:,:,:].cpu().numpy()
        vol_mask_array = batch['label'][0,0,:,:,:].cpu().numpy().astype(np.uint8)
        #taglio la segmentazione automatica sempre rispetto al centroide calcolato con maschera manuale, così combaciano sicuro
        vol_pred_array = batch['pred'][0,0,:,:,:].cpu().numpy().astype(np.uint8) 
        #inizializzo le coordinate del centro con il generico centro dell'immagine
        #per poi prendendere il centroide della porzione che contiene l'oggetto
        x_c = [vol_array.shape[1]/2]
        y_c = [vol_mask_array.shape[0]/2]
        
        perc_1,perc_99 = percentile(vol_array,vol_mask_array)
        
        n_slices = vol_array.shape[2] #--> potevo prendere indifferentemente image_array o mask_array perchè hanno lo stesso numero di slices
        
#-------------se voglio le immagini divise per cartelle-paziente---------------
        # if not os.path.isdir(os.path.join(img_dir,image_id)):
        #      os.mkdir(os.path.join(img_dir,image_id))
        # if not os.path.isdir(os.path.join(mask_dir,image_id)):
        #      os.mkdir(os.path.join(mask_dir,image_id))
#------------------------------------------------------------------------------        
        for n in range(n_slices):
            if n == 0 or n % 3 == 0 :
                mask_array = vol_mask_array[:,:,n]
                pred_array = vol_pred_array[:,:,n]
                img_array = vol_array[:,:,n]
                
                
                try:
                    img_patch,mask_patch = centered_crop(img_array,mask_array,pred_array, width, height,x_c,y_c)

                    #normalizzo rispetto al 1° e 99° perc l'immagine croppata
                    slice_adj = imadjust(img_patch,perc_1,perc_99)
                    cv2.imwrite(os.path.join(img_dir,'{}_{}.png'.format(image_id,n)),slice_adj)
                    cv2.imwrite(os.path.join(mask_dir,'{}_{}.png'.format(image_id,n)),mask_patch)  
                except:
                    print(os.path.join(img_dir,'{}_{}_mask.png'.format(image_id,n)))
                
              
#%%

#coscia 

make_slices(test_loader_c,img_coscia_TS,mask_coscia_TS,176,176)

#gamba 

make_slices(test_loader_g,img_gamba_TS,mask_gamba_TS,160,160)


#%% selezionare solo le slice match
#COMMENTARE E DECOMMENTARE OPPORTUNAMENTE LE CARTELLE

#mask_dir = mask_coscia_TS
mask_dir = mask_gamba_TS

#directory = 'C:/Users/omobo/Desktop/COSCIA'
directory = 'C:/Users/omobo/Desktop/GAMBA'

#MASK_path = sorted(glob(os.path.join(directory,'outputsTs-coscia','*.png')))  #PER ESTRARRE LE SLICE DALLE PREDICTION
LABEL_path = sorted(glob(os.path.join(directory,'labels','*.png')))  #PER ESTRARRE LE SLICE DA GROUND TRUTH

for l in LABEL_path:
    #img = cv2.imread(i)
    label_name = os.path.basename(l)
    print(label_name)
    try:
        file_path = os.path.join(mask_dir,label_name)
        #dest = os.path.join(directory,'output_swin_3D')
        dest = os.path.join(directory,'output_Unet_3D')
        shutil.move(file_path,dest)
    except:
        print('file already moved!')