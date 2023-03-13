
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:46:53 2023

@author: omobo
"""


"""
PRENDO LE SEGMENTAZIONI AUTOMATICHE 2D
APPLICO IL POST PROCESSING
RIPRISTINO LE DIMENSIONI DELLE SLICE:
    160X200 o 256x192
RICREO I VOLUMI
"""
"""
PRENDO TUTTE LE PATCH SPARSE
LE ORDINO IN CARTELLE CHE HANNO LO STESSO NOME DEL "VOLUME"
ad esempio: metto tutte le slice {0_005_1_1_00.png,0_005_1_1_01.png,...,0_005_1_1_31.png} nella cartella "0_005_1_1"
DOPO, MANUALMENTE, METTO IL FILE .nii CORRISPONDENTE NELLE CARTELLE (ad esempio 0_005_1_1_.nii nella cartella 0_005_1_1)
QUESTO SERVE PER AVERE IL CORRETTO VOLUME DI RIFERIMENTO ALL'INTERNO DELLA CARTELLA STESSA, COSì IL CODICE è PIù SEMPLICE

"""

import numpy as np

import os
from glob import glob
from skimage.morphology import remove_small_objects,label, area_closing
from skimage import segmentation

import SimpleITK as sitk

import cv2
import shutil

#%% CARTELLE UTILI

# directory= 'C:/Users/omobo/Desktop/all image/GAMBE'

# in_images = os.path.join(directory,'imagesTs-gamba')
# in_labels = os.path.join(directory,'labelsTs-gamba')
# in_preds = os.path.join(directory,'outputsTs-gamba')


# outputs_directory = "C:/Users/omobo/Desktop/VOLUMI/GAMBA"

# out_images = os.path.join(outputs_directory,'images')
# out_labels = os.path.join(outputs_directory,'labels')
# #out_preds = os.path.join(outputs_directory,'outputs')


# images_path = sorted(glob(os.path.join(in_images,'*.png')))
# labels_path = sorted(glob(os.path.join(in_labels,'*.png')))

in_labels_c_r = "C:/Users/omobo/Desktop/COSCIA/RESULTS_RESNET/test_all/mask"
out_labels_c_r = "C:/Users/omobo/Desktop/VOLUMI/COSCIA/mask_resnet"

in_labels_c_s = "C:/Users/omobo/Desktop/COSCIA/RESULTS_SWIN/test_all/mask"
out_labels_c_s = "C:/Users/omobo/Desktop/VOLUMI/COSCIA/mask_swin"

in_labels_g_r = "C:/Users/omobo/Desktop/GAMBA/RESULTS_RESNET/test_all/mask"
out_labels_g_r = "C:/Users/omobo/Desktop/VOLUMI/GAMBA/mask_resnet"

in_labels_g_s = "C:/Users/omobo/Desktop/GAMBA/RESULTS_SWIN/test_all/mask"
out_labels_g_s = "C:/Users/omobo/Desktop/VOLUMI/GAMBA/mask_swin"


labels_path_c_r = sorted(glob(os.path.join(in_labels_c_r,'*.png')))
labels_path_c_s = sorted(glob(os.path.join(in_labels_c_s,'*.png')))
labels_path_g_r = sorted(glob(os.path.join(in_labels_g_r,'*.png')))
labels_path_g_s = sorted(glob(os.path.join(in_labels_g_s,'*.png')))
#-----------------------------------------------------------------------
"""
FUNZIONE : rename

da:
    0_005_1_1_0.png
    0_005_1_1_1.png
    0_005_1_1_2.png
    0_005_1_1_3.png
    ....
    0_005_1_1_30.png
    0_005_1_1_31.png
    
a:
    0_005_1_1_00.png
    0_005_1_1_01.png
    0_005_1_1_02.png
    0_005_1_1_03.png
    ....
    0_005_1_1_30.png
    0_005_1_1_31.png
    
così le riordina correttamente senza sfasarle. (con la numerazione precedente venivano sfasate)
"""
lista = []
for i in range(10):
    lista.append(str(i))

def rename (path):
    for i in path:
        image_name = os.path.splitext(i)[0]
        num_slice = image_name.split('_')[-1]
        if num_slice in lista:
                new_i = i.replace('{}.png'.format(num_slice),'0{}.png'.format(num_slice))
                print(new_i)
                os.rename(i,new_i)
                new_mask_name = os.path.basename(new_i)
                print('new mask name : ', new_mask_name)
           
    
rename(labels_path_c_r)
rename(labels_path_c_s)
rename(labels_path_g_r)
rename(labels_path_g_s)

#%%
#CREO LE CARTELLE PAZIENTE

def make_vol_dir(in_dir):
    path = sorted(glob(os.path.join(in_dir,'*.png')))
    list_id = []
    for i in path:
        image_name = os.path.basename(i)
        image_name = image_name.split('_')
        image_name = image_name[0:-1]
        image_id = '_'.join(image_name)
        if image_id not in list_id:
            list_id.append(image_id)
            

    for i in list_id:
        vol_dir = os.path.join(in_dir,'{}'.format(i))
        if not os.path.exists(vol_dir):
          os.mkdir(vol_dir)
          
    for i in path:
        image_name = os.path.basename(i)
        image_name = image_name.split('_')
        image_name = image_name[0:-1]
        image_id = '_'.join(image_name)
        
        dest = os.path.join(in_dir,'{}'.format(image_id))
        shutil.move(i,dest)

make_vol_dir(in_labels_c_r)    
make_vol_dir(in_labels_c_s)
make_vol_dir(in_labels_g_r)
make_vol_dir(in_labels_g_s)    


    
    

#%%
#le cartelle devono contenere le patches da ricomporre nel volume e il volume di riferimento

def make_volumes(in_dir,out_dir,pad):
    all_dir = sorted(glob(os.path.join(in_dir,'*'))) 
      
        
    for path in all_dir:   #--> path = "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1"
        
        vol_path = glob(os.path.join(path,'*.nii'))  #--> "C:\Users\omobo\Desktop\all image\COSCE\labelsTs-coscia\0_005_1_1\0_005_1_1_mask.nii"
        vol_name = os.path.basename(vol_path[0])  #-->"0_005_1_1_mask.nii"
        print(vol_name)
        original_volume = sitk.ReadImage(vol_path[0])
        or_vol_array = sitk.GetArrayFromImage(original_volume)
        
        num_slices = or_vol_array.shape[0]  #--> 32
        width = or_vol_array.shape[1]  #-->192
        height = or_vol_array.shape[2] #-->256
        
        images = sorted(glob(os.path.join(path,'*.png')))
        
        volume_array = np.zeros((height,width,num_slices))
        
        
        for n,img in enumerate(images):
            
            a = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            # plt.figure("check", (12, 6))
            # plt.subplot(1, 1, 1)
            # plt.title("before_post_proc_slice: {}".format(os.path.basename(img)))
            # plt.imshow(a.astype(np.uint8), cmap="nipy_spectral",interpolation='nearest')
            # plt.show()
            
            data_array = a.astype(np.uint8)
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
            #------------------------------------------------------------------
            
            #zeropadding
            b = np.pad(post_proc_slice,((pad,pad),(pad,pad))).astype(np.uint8)
            #plt.imshow(b)
            
            x_c = b.shape[1]/2
            y_c = b.shape[0]/2
            ymin = int(y_c - height/2) 
            ymax = int(y_c + height/2) 
            xmin = int(x_c - width/2) 
            xmax = int(x_c + width/2) 
            
            #crop
            img_patch = (b[ymin:ymax,xmin:xmax]).astype(np.uint8)
            
            #mask_patch = mask_array[ymin:ymax,xmin:xmax]
            
            volume_array[:,:,n] = img_patch
        
        # salvo nuovo volume
        volume_array_T = volume_array.T
        volume_array_T =  volume_array_T.astype(np.uint16)   
        newVolume = sitk.GetImageFromArray(volume_array_T)
        newVolume.CopyInformation(original_volume)
        sitk.WriteImage(newVolume, os.path.join(out_dir,vol_name))


    
    
        
#COSCIA        
make_volumes(in_labels_c_r,out_labels_c_r,60)        
make_volumes(in_labels_c_s,out_labels_c_s,60)             

#GAMBA
make_volumes(in_labels_g_r,out_labels_g_r,100)        
make_volumes(in_labels_g_s,out_labels_g_s,100)    
        
