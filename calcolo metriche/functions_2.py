# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:39:44 2023

@author: omobo
"""


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.morphology import remove_small_objects,label, area_closing
from skimage import segmentation
from glob import glob

from monai.data import  decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import torch
import pandas as pd


#%%
#==============================================================================
#                      INFERENCE, POST-PROCESSING E SALVATAGGIO
#==============================================================================

"""
Questa funzione calcola l'inference ,il post-process sui volumi di output
e, in fine, salva le maschere automatiche post-processate

"""

def save_post_proc(loader, post_pred, output_dir, device, model):
    print("------------------------------")
    with torch.no_grad():
        
            for batch in loader:  
                inputs = batch["image"].to(device)
                img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])   #-->2_023_1_2.nii
                patient = img_name.split('_')[1]  #--> 023
                #affine: sulle label non ho trasformato lo spacing--> posso ricavare
                #l'affine originale e salvare i volumi in formato .nii 
                affine = batch['label_meta_dict']['original_affine'][0].numpy()   
                print("Inference on patient {}".format(patient))
                batch["pred"] = sliding_window_inference(inputs, (96, 96, 32), 2, model)

                batch = [post_pred(i) for i in decollate_batch(batch)] #batch adesso è una lista
                #ho ottenuto l'output con lo spacing originale, in funzione del post-processing            
                outputs = (batch[0]["pred"].to(device)).cpu().numpy()[0]  #-->ottengo un array 3D
                
                print('Post-processing')
                post_auto_mask = np.zeros_like(outputs)
                for n_slice in range(outputs.shape[2]):  #voglio prendere tutte le slices del volume corrente
                    data_array = outputs[:,:,n_slice].astype(np.uint8)
                    #------------------------------------------------------------------                    
                    # remove small objects
                    array = label(data_array)
                    obj_removed = remove_small_objects (array , min_size = 60) 
                    object_removed = segmentation.watershed(
                              obj_removed, data_array, mask= obj_removed)
                    
                    # fill holes
                    post_proc_slice = area_closing(object_removed, area_threshold = 40)
                    #------------------------------------------------------------------
                    #concateno la slice corrente alle precedenti per ottenere di nuovo il volume
                    post_auto_mask[:,:,n_slice] = post_proc_slice
                
                #converto l'array 3D in un file nifti per poi salvarlo
                final_vol = nib.Nifti1Image(post_auto_mask, affine)
                nib.save(final_vol,os.path.join(output_dir,'{}_pred.nii'.format(img_name.split('.')[0])))
                print('Image {}_pred.nii Saved!'.format(img_name.split('.')[0]))
            

    
#%%
"""
creare l' "impalcatura" dei dictionaries che conterranno le metriche.
Consideriamo il Validation Set. Un esempio di quello che otteniamo è:

VAL = {}
VAL["FSHD"] = {}
VAL["MD1"] = {}

VAL["FSHD"]["008"] = {}
VAL["FSHD"]["010"] = {}
VAL["FSHD"]["013"] = {}
VAL["FSHD"]["008"]["L"] = {}
VAL["FSHD"]["008"]["R"] = {}
VAL["FSHD"]["010"]["L"] = {}
VAL["FSHD"]["010"]["R"] = {}
VAL["FSHD"]["013"]["L"] = {}
VAL["FSHD"]["013"]["R"] = {}

VAL["MD1"]["008"] = {}
VAL["MD1"]["010"] = {}
VAL["MD1"]["023"] = {}
VAL["MD1"]["008"]["1"] = {}
VAL["MD1"]["008"]["2"] = {}
VAL["MD1"]["008"]["3"] = {}
VAL["MD1"]["010"]["1"] = {}
VAL["MD1"]["023"]["1"] = {}
VAL["MD1"]["023"]["2"] = {}
"""

def make_dict(loader,dict_name):

    #questa funzione può essere divisa concettualmente in due parti:
    #--------------------------------------------------------------------------    
    #------------------------------PRIMA PARTE---------------------------------
    #in questa prima parte vengono create delle liste utili a creare automaticamente i dictionaries che 
    #ospiteranno le metriche suddivise in subsets.
    
    patient_type_list = [] #--> ["1","2"] ad esempio,per il Validation. Indica il tipo di pazienti contenuti nel set
    patient_0 = []  #-->[]  Pazienti di tipo 0 (HV)
    patient_1 = []  #-->["008","010","013"]   Pazienti di tipo 1 (FSHD)
    patient_2 = []  #-->["008","010","023"]   Pazienti di tipo 2 (MD1)
    
    # ad alcuni pazienti di tipo 2 sono associati 3 volumi, ad altri 1, ad altri ancora 2.
    #per gestire le varie casistiche e automatizzare il processo sono state create list_2 e list_3
    #lista_provv serve per creare lista_2
    
    list_provv = [] #-->["008","023"]
    list_2 = [] #-->["023"]   Lista pazienti con 2 volumi
    list_3 = [] #-->["008"]   Lista pazienti con 3 volumi

    for batch in loader:
        img_name = os.path.basename(batch["label_meta_dict"]["filename_or_obj"][0])   #--> "2_023_1_1.nii"
        patient_type = img_name.split("_")[0] #-->" 2"
        patient = img_name.split("_")[1]  #--> "023"
        side_or_num = os.path.splitext(img_name)[0].split("_")[-2]    #--> può essere: "L","R", "1", "2", "3"
        
        if patient_type == "0":
            if patient not in patient_0:
                patient_0.append(patient)
        
        if patient_type == "1":
            if patient not in patient_1:
                patient_1.append(patient)
        
        if patient_type == "2":
            if patient not in patient_2:
                patient_2.append(patient)
            
            
        if patient_type not in patient_type_list:
            patient_type_list.append(patient_type)
            
        if side_or_num == "2":
            list_provv.append(patient)    
        
        if side_or_num == "3":
            list_3.append(patient)
            
        
    for i in list_provv:
        if i in list_provv and i not in list_3:
            list_2.append(i)
    #--------------------------------------------------------------------------
    #----------------------------SECONDA PARTE---------------------------------
    #creazione dei dictionaries
        
    if "0" in patient_type_list:
        dict_name["HV"] = {}
        for i in patient_0:
            dict_name["HV"][i] = {}
            #decommentare (e ripetere negli altri casi) nel caso di DATASET unico
            # for j in ["Leg","Thigh"]:
            #     dict_name["HV"][i][j] = {}
            
    if "1" in patient_type_list:
        dict_name["FSHD"] = {}
        for i in patient_1:
            dict_name["FSHD"][i] = {}
            for k in ["L","R"]:
                dict_name["FSHD"][i][k] = {}
                
    if "2" in patient_type_list:
        dict_name["MD1"] = {}
        for i in patient_2:
            dict_name["MD1"][i] = {}
            if i in list_3:
                for k in ["1","2","3"]:
                    dict_name["MD1"][i][k] = {}
            if i in list_2:
                for k in ["1","2"]:
                    dict_name["MD1"][i][k] = {}
            else:
                dict_name["MD1"][i]["1"] = {}
    return dict_name

    
#%%

def calc_metrics(loader,device ,post_pred , dict_name, muscles):


    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    hd95_metric_batch = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    print("------------------------------")
    with torch.no_grad():
            dice_vals = []
            dice_vals_batch = []
            hd95_vals = []
            hd95_vals_batch = []
            for batch in loader:  
                pred, labels = (batch["pred"].to(device), batch["label"].to(device))
                img_name = os.path.basename(batch["label_meta_dict"]["filename_or_obj"][0])   #-->2_023_1_2.nii
                patient_type = img_name.split('_')[0]  #--> 2
                patient = img_name.split('_')[1]  #--> 023
                
                #val_outputs = torch.softmax(val_outputs,1)

                pred = [post_pred(d) for d in decollate_batch(pred)]
                labels = [post_pred(d) for d in decollate_batch(labels)]

    #------------------------CALCOLO METRICHE: DICE E HD95---------------------
    #calcolo le metriche per ogni label di ogni soggetto del set

                dice_metric(y_pred=pred, y=labels)
                dice_metric_batch(y_pred=pred, y=labels)
                hd95_metric(y_pred = pred, y = labels)
                hd95_metric_batch(y_pred = pred, y = labels)
    #--------------------------------------------------------------------------
    #alla fine, intendo calcolare uno score complessivo (ogni label di ogni paziente)#------> non lo faccio più
    
                #metriche: all_label_all_patient
                dice = dice_metric.aggregate().item()  
                dice_vals.append(dice)
                
                hd95 = hd95_metric.aggregate().item()  
                hd95_vals.append(hd95)

    #--------------------------------------------------------------------------
    #per ogni paziente otteniamo il dice/hd95 di ogni label --> 13 valori (ad ogni iterazione)            
                #metriche: single_label_single_patient
                dice_batch = dice_metric_batch.aggregate().cpu().numpy()  
                hd95_batch = hd95_metric_batch.aggregate().cpu().numpy()
                
                dice_vals_batch.append(dice_batch) # Ogni elemento è un array 1x13.
                hd95_vals_batch.append(hd95_batch)
                
    #inserisco le metriche di ogni label per ogni paziente nell'apposito dictionary
    #in cui si tiene conto del tipo del soggetto.
    #Vengono escluse le metriche dello sfondo.
    
                if patient_type == "0":
                    dict_name["HV"][patient]["dice"] = {muscles[i-1]:dice_batch[i] for i in range(1,len(dice_batch))}
                    dict_name["HV"][patient]["hd95"] = {muscles[i-1]:hd95_batch[i] for i in range(1,len(hd95_batch))}
            
                if patient_type == "1":
                    side = img_name.split('_')[4]
                    dict_name["FSHD"][patient][side]["dice"] = {muscles[i-1]:dice_batch[i] for i in range(1,len(dice_batch))}
                    dict_name["FSHD"][patient][side]["hd95"] = {muscles[i-1]:hd95_batch[i] for i in range(1,len(hd95_batch))}
            
                        
                if patient_type == "2" :
                    k = img_name.split('_')[3]
                    dict_name["MD1"][patient][k]["dice"] = {muscles[i-1]:dice_batch[i] for i in range(1,len(dice_batch))}
                    dict_name["MD1"][patient][k]["hd95"] = {muscles[i-1]:hd95_batch[i] for i in range(1,len(hd95_batch))}
          
    
    return dict_name , dice_vals_batch , hd95_vals_batch , dice_vals , hd95_vals


#%%

def mean_muscle_metrics(dice_vals_batch, hd95_vals_batch, dict_name, muscles):
    mean_muscle_dice = []
    mean_muscle_hd95 = []
    for j in range(1,len(dice_vals_batch[0])):# j va da 1 a 13 = prendo solo il dice dei muscoli . 
        muscle_dice = []
        for i in range(len(dice_vals_batch)):  #i va da 0 a 11 = 12 elementi del validation
            #m_d sta per muscle dice
            m_d = dice_vals_batch[i][j] #prende i dice calcolati su ogni paziente per quel singolo muscolo
            muscle_dice.append(m_d)
        mean_muscle = np.mean(muscle_dice)  #ottengo la media del dice del singolo muscolo 
        mean_muscle_dice.append(mean_muscle)
        
    for j in range(1,len(hd95_vals_batch[0])):# j va da 1 a 13 = prendo solo il dice dei muscoli . 
        muscle_hd95 = []
        for i in range(len(hd95_vals_batch)):  
            m_hd95 = hd95_vals_batch[i][j] 
            muscle_hd95.append(m_hd95)
        mean_muscle = np.mean(muscle_hd95)  #ottengo la media del dice del singolo muscolo 
        mean_muscle_hd95.append(mean_muscle)
    
    dict_name = {}
    dict_name["dice"] = {muscles[i]:mean_muscle_dice[i] for i in range(len(mean_muscle_dice))} 
    dict_name["hd95"] = {muscles[i]:mean_muscle_hd95[i] for i in range(len(mean_muscle_hd95))}

    return dict_name   

#%%

def std_muscle_metrics(dice_vals_batch, hd95_vals_batch, dict_name, muscles):
    mean_muscle_dice = []
    mean_muscle_hd95 = []
    for j in range(1,len(dice_vals_batch[0])):# j va da 1 a 13 = prendo solo il dice dei muscoli . 
        muscle_dice = []
        for i in range(len(dice_vals_batch)):  #i va da 0 a 11 = 12 elementi del validation
            #m_d sta per muscle dice
            m_d = dice_vals_batch[i][j] #prende i dice calcolati su ogni paziente per quel singolo muscolo
            muscle_dice.append(m_d)
#------------------------------------------------------------------------------        
        mean_muscle = np.std(muscle_dice)  #ottengo la STANDARD DEV del dice del singolo muscolo 
#------------------------------------------------------------------------------        
        mean_muscle_dice.append(mean_muscle)
        
    for j in range(1,len(hd95_vals_batch[0])):# j va da 1 a 13 = prendo solo il dice dei muscoli . 
        muscle_hd95 = []
        for i in range(len(hd95_vals_batch)):  
            m_hd95 = hd95_vals_batch[i][j] 
            muscle_hd95.append(m_hd95)
#------------------------------------------------------------------------------            
        mean_muscle = np.std(muscle_hd95)  #ottengo la media del dice del singolo muscolo 
#------------------------------------------------------------------------------
        mean_muscle_hd95.append(mean_muscle)
    
    dict_name = {}
    dict_name["dice"] = {muscles[i]:mean_muscle_dice[i] for i in range(len(mean_muscle_dice))} 
    dict_name["hd95"] = {muscles[i]:mean_muscle_hd95[i] for i in range(len(mean_muscle_hd95))}

    return dict_name 
#%%

# def plot_results(labels,muscles,muscle_metric_TEST,muscle_std_TEST, metric , title):
#     #labels = è una lista contenente le abbreviazioni dei nomi dei muscoli
#     #muscles = è la lista che contiene le chiavi per i dictionaries delle metriche
#     #muscle_metric_TRAIN(VAL/TEST) sono le metriche da plottare
#     #mean_train(val/test) sono gli overall scores
#     #metric--> può essere 'dice' o 'hd95' a seconda della metrica che vogliamo plottare
#     #title --> indica le metriche plottate 
    
#     data = []
#     for i in range(len(labels)):
#       metric_set = [labels[i],
#                 muscle_metric_TEST[metric][muscles[i]],
#                 muscle_std_TEST[metric][muscles[i]]]
      
#       data.append(metric_set)
      
#     #
#     df_provv=pd.DataFrame(data,columns=["Muscles", "TEST","std"])
    
 
#     mean_test = np.mean(df_provv['TEST'].values)
#     std_test = np.mean(df_provv['std'].values)
    
#     avg_dice = ["All Muscle Avg",mean_test,std_test]
#     data.append(avg_dice)
    
    
    
#     df=pd.DataFrame(data,columns=["Muscles", "TEST","std"])

#     print(title,'\n',df)
#     df.plot(x="Muscles", y=["TEST","std"], kind="bar",figsize=(12,16),title=title, legend = True)
#     plt.show()

def plot_results(labels,muscles,muscle_metric_TEST,muscle_std_TEST, metric , title):
    #labels = è una lista contenente le abbreviazioni dei nomi dei muscoli
    #muscles = è la lista che contiene le chiavi per i dictionaries delle metriche
    #muscle_metric_TRAIN(VAL/TEST) sono le metriche da plottare
    #mean_train(val/test) sono gli overall scores
    #metric--> può essere 'dice' o 'hd95' a seconda della metrica che vogliamo plottare
    #title --> indica le metriche plottate 
    
    data = []
    std = []
    mean = []
    for i in range(len(labels)):
        
      if metric == 'dice':
          a = round(muscle_metric_TEST[metric][muscles[i]]*100,2) 
          b = round(muscle_std_TEST[metric][muscles[i]]*100,2)
      else:
          a = round(muscle_metric_TEST[metric][muscles[i]],2) 
          b = round(muscle_std_TEST[metric][muscles[i]],2)
          
      
      c = (str(a)+' ± '+str(b))
      metric_set = [labels[i],c]
      
      data.append(metric_set)
      std.append(b)
      mean.append(a)
    #
    #df_provv=pd.DataFrame(data,columns=["Muscles", "TEST"])
    
 
    mean_test = np.mean(mean)
    std_test = np.std(mean)
    
    avg_dice = ["All Muscle Avg",(str(round(mean_test,2))+' ± '+str(round(std_test,2)))]
    data.append(avg_dice)
    
    
    df_read=pd.DataFrame(data,columns=["Muscles", "TEST"])

    print(title)
    for num in df_read["TEST"].values:
        print(num)
    print('  ')
    data = []
    for i in range(len(labels)):
      metric_set = [labels[i],
                muscle_metric_TEST[metric][muscles[i]],
                muscle_std_TEST[metric][muscles[i]]]
      
      data.append(metric_set)
      
    #
    df_provv=pd.DataFrame(data,columns=["Muscles", "TEST","std"])
    
 
    mean_test = np.mean(df_provv['TEST'].values)
    std_test = np.std(df_provv['TEST'].values)
    
    avg_dice = ["All Muscle Avg",mean_test,std_test]
    data.append(avg_dice)
    
    
    
    df=pd.DataFrame(data,columns=["Muscles", "TEST","std"])
    
    
    df.plot(x="Muscles", y=["TEST"], kind="bar",figsize=(10,7),title=title,fontsize =10,yerr="std",capsize=8)
    plt.show()