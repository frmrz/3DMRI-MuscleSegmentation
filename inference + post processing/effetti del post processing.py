# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:32:34 2023

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


path_1 = "C:/Users/omobo/Desktop/post-proc_hd95.csv"
path_2 = "C:/Users/omobo/Desktop/post-proc_dsc.csv"
path_3 = "C:/Users/omobo/Desktop/avg_hd95.csv"
path_4 = "C:/Users/omobo/Desktop/avg_dsc.csv"


df_1 = pd.read_csv(path_1,index_col=0)
df_2 = pd.read_csv(path_2,index_col=0)
df_3 = pd.read_csv(path_3,index_col=0)
df_4 = pd.read_csv(path_4,index_col=0)

# df_1.plot(x="Muscle", y=['p-p 0','p-p 1','p-p 2','p-p 3'], kind="bar",figsize=(10,7),title="HD95",legend = True, fontsize =10,yerr=['std 0','std 1','std 2','std 3'],capsize=8)
# plt.show()

# data=df_1[['Muscle','std 0','std 1','std 2','std 3']].values
# std_1=pd.DataFrame(data,columns=['Muscle','p-p 0','p-p 1','p-p 2','p-p 3'])
# std_1.to_csv("C:/Users/omobo/Desktop/std_hd95.csv")
std_1 = pd.read_csv("C:/Users/omobo/Desktop/std_hd95.csv",index_col=0)

# data=df_2[['Muscle','std 0','std 1','std 2','std 3']].values
# std_2=pd.DataFrame(data,columns=['Muscle','p-p 0','p-p 1','p-p 2','p-p 3'])
# std_2.to_csv("C:/Users/omobo/Desktop/std_dsc.csv")
std_2 = pd.read_csv("C:/Users/omobo/Desktop/std_dsc.csv",index_col=0)

# data=df_3[['std 0','std 1','std 2','std 3']].values
# std_3=pd.DataFrame(data,columns=['p-p 0','p-p 1','p-p 2','p-p 3'])
# std_3.to_csv("C:/Users/omobo/Desktop/std_avg_hd95.csv")
std_3 = pd.read_csv("C:/Users/omobo/Desktop/std_avg_hd95.csv",index_col=0)

# data=df_4[['std 0','std 1','std 2','std 3']].values
# std_4=pd.DataFrame(data,columns=['p-p 0','p-p 1','p-p 2','p-p 3'])
# std_4.to_csv("C:/Users/omobo/Desktop/std_avg_dsc.csv")
std_4 = pd.read_csv("C:/Users/omobo/Desktop/std_avg_dsc.csv",index_col=0)






data=df_1[['Muscle','p-p 0','p-p 1','p-p 2','p-p 3']].values
df_1=pd.DataFrame(data,columns=['Muscle','p-p 0','p-p 1','p-p 2','p-p 3'])
df_1.to_csv(path_1)
df_1 = pd.read_csv(path_1,index_col=0)


data=df_2[['Muscle','p-p 0','p-p 1','p-p 2','p-p 3']].values
df_2=pd.DataFrame(data,columns=['Muscle','p-p 0','p-p 1','p-p 2','p-p 3'])
df_2.to_csv(path_2)
df_2 = pd.read_csv(path_2,index_col=0)


data=df_3[['p-p 0','p-p 1','p-p 2','p-p 3']].values
df_3=pd.DataFrame(data,columns=['p-p 0','p-p 1','p-p 2','p-p 3'])
df_3.to_csv(path_3)
df_3 = pd.read_csv(path_3,index_col=0)


data=df_4[['p-p 0','p-p 1','p-p 2','p-p 3']].values
df_4=pd.DataFrame(data,columns=['p-p 0','p-p 1','p-p 2','p-p 3'])
df_4.to_csv(path_4)
df_4 = pd.read_csv(path_4,index_col=0)


# df_1.plot(x="Muscle", y=['p-p 0','p-p 1','p-p 2','p-p 3'], kind="bar",figsize=(10,7),title="HD95",legend = True, fontsize =10, yerr = std_1, capsize=12)
# plt.show()

# ticks = ['TA','TP ', 'ES-L','PER','FL-D','SOL','GM','GL','FL-A']



fig,ax=plt.subplots()
df_2.plot.bar(figsize=(10,7),title="DSC",legend = True, fontsize =15,colormap='cool',yerr=std_2,ax=ax,capsize=3,rot = 0)


fig,ax=plt.subplots()
df_1.plot.bar(figsize=(10,7),title="HD95",legend = True, fontsize =15,colormap='summer',yerr=std_1,ax=ax,capsize=3,rot = 0)


fig,ax=plt.subplots()
df_4.plot.bar(figsize=(10,7),title="DSC MEDIO",legend = True,xlabel='confronto complessivo tra i post-processig', fontsize =15,colormap='cool',yerr=std_4,ax=ax,capsize=10,rot = 0)


fig,ax=plt.subplots()
df_3.plot.bar(figsize=(10,7),title="HD95 MEDIO",legend = True,xlabel='confronto complessivo tra i post-processig', fontsize =15,colormap='summer',yerr=std_3,ax=ax,capsize=10,rot = 0)
