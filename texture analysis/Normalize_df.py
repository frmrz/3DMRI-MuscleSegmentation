# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 09:30:22 2023

@author: omobo
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor

import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from glob import glob
import pandas as pd


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from statsmodels.multivariate.manova import MANOVA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#from umap import UMAP

import plotly.express as px
import plotly.io as pio


"""normalizzo il dataframe con min max scaler"""
directory = "C:/Users/omobo/Desktop"
df_norm_dir = os.path.join(directory,'DataframeNormalizzati')
if not os.path.exists(df_norm_dir):
  os.mkdir(df_norm_dir) 
  
df_norm_list = []


#io ho i df non normalizzati completi nella cartella dataframe del desktop
#devo prendere quelli, normalizzarli e salvarli nella cartella df normalizzati

df_dir =  "C:/Users/omobo/Desktop/DataframeVM"
df_paths = glob(os.path.join(df_dir,'*.csv'))


#------------------------------------------------------------------------------


feature_names = ['original_firstorder_10Percentile','original_firstorder_90Percentile','original_firstorder_Energy',
'original_firstorder_Entropy' ,
'original_firstorder_InterquartileRange' ,
'original_firstorder_Kurtosis' ,
'original_firstorder_Maximum' ,
'original_firstorder_Mean' ,
'original_firstorder_MeanAbsoluteDeviation' ,
'original_firstorder_Median' ,
'original_firstorder_Minimum' ,
'original_firstorder_Range' ,
'original_firstorder_RobustMeanAbsoluteDeviation',
'original_firstorder_RootMeanSquared' ,
'original_firstorder_Skewness' ,
'original_firstorder_TotalEnergy' ,
'original_firstorder_Uniformity' ,
'original_firstorder_Variance',
'original_glcm_Autocorrelation' ,
'original_glcm_ClusterProminence' ,
'original_glcm_ClusterShade' ,
'original_glcm_ClusterTendency' ,
'original_glcm_Contrast' ,
'original_glcm_Correlation' ,
'original_glcm_DifferenceAverage' ,
'original_glcm_DifferenceEntropy' ,
'original_glcm_DifferenceVariance',
'original_glcm_Id' ,
'original_glcm_Idm' ,
'original_glcm_Idmn',
'original_glcm_Idn' ,
'original_glcm_Imc1' ,
'original_glcm_Imc2',
'original_glcm_InverseVariance',
'original_glcm_JointAverage' ,
'original_glcm_JointEnergy',
'original_glcm_JointEntropy',
'original_glcm_MaximumProbability',
'original_glcm_SumEntropy' ,
'original_glcm_SumSquares',
'original_gldm_DependenceEntropy',
'original_gldm_DependenceNonUniformity',
'original_gldm_DependenceNonUniformityNormalized',
'original_gldm_DependenceVariance',
'original_gldm_GrayLevelNonUniformity',
'original_gldm_GrayLevelVariance',
'original_gldm_HighGrayLevelEmphasis',
'original_gldm_LargeDependenceEmphasis',
'original_gldm_LargeDependenceHighGrayLevelEmphasis',
'original_gldm_LargeDependenceLowGrayLevelEmphasis',
'original_gldm_LowGrayLevelEmphasis' ,
'original_gldm_SmallDependenceEmphasis',
'original_gldm_SmallDependenceHighGrayLevelEmphasis',
'original_gldm_SmallDependenceLowGrayLevelEmphasis',
'original_glrlm_GrayLevelNonUniformity',
'original_glrlm_GrayLevelNonUniformityNormalized',
'original_glrlm_GrayLevelVariance',
'original_glrlm_HighGrayLevelRunEmphasis',
'original_glrlm_LongRunEmphasis',
'original_glrlm_LongRunHighGrayLevelEmphasis',
'original_glrlm_LongRunLowGrayLevelEmphasis',
'original_glrlm_LowGrayLevelRunEmphasis',
'original_glrlm_RunEntropy',
'original_glrlm_RunLengthNonUniformity',
'original_glrlm_RunLengthNonUniformityNormalized',
'original_glrlm_RunPercentage',
'original_glrlm_RunVariance',
'original_glrlm_ShortRunEmphasis',
'original_glrlm_ShortRunHighGrayLevelEmphasis',
'original_glrlm_ShortRunLowGrayLevelEmphasis',
'original_glszm_GrayLevelNonUniformity',
'original_glszm_GrayLevelNonUniformityNormalized',
'original_glszm_GrayLevelVariance' ,
'original_glszm_HighGrayLevelZoneEmphasis',
'original_glszm_LargeAreaEmphasis',
'original_glszm_LargeAreaHighGrayLevelEmphasis',
'original_glszm_LargeAreaLowGrayLevelEmphasis' ,
'original_glszm_LowGrayLevelZoneEmphasis',
'original_glszm_SizeZoneNonUniformity' ,
'original_glszm_SizeZoneNonUniformityNormalized' ,
'original_glszm_SmallAreaEmphasis' ,
'original_glszm_SmallAreaHighGrayLevelEmphasis' ,
'original_glszm_SmallAreaLowGrayLevelEmphasis' ,
'original_glszm_ZoneEntropy' ,
'original_glszm_ZonePercentage' , 'original_glszm_ZoneVariance']

columns = ['Subject']
for feature in feature_names:
    columns.append(feature)
columns.append('Class')


for path in df_paths:
    label = os.path.splitext(path)[0].split('_')[1]  #-->'VL'
    id_ = os.path.splitext(path)[0].split('_')[2]
    df = pd.read_csv(path, index_col=0)
    data_subset = df[feature_names].values
    data_subset_scaled = MinMaxScaler().fit_transform(data_subset)
    df_subject= list(df['Subject'])
    df_classes = list(df['Class'])
    new_data =[]
    for i in range(len(df_subject)):
        row = [df_subject[i]]
        for j in range(data_subset_scaled.shape[1]):
            features_norm = data_subset_scaled[i,j]
            row.append(features_norm)
        
        class_id = df_classes[i]
        row.append(class_id)
        new_data.append(row)   #5x88
    #new_df = [df_subject,data_subset_scaled,df_classes]
    df_norm = pd.DataFrame(new_data,columns=columns)
    df_norm.to_csv(os.path.join(df_dir,'Dataframe_{}_{}.csv'.format(label,id_)))
    df_norm_list.append(df_norm)
