# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:58:05 2023

@author: omobo
"""


import os

from scipy import stats 
import matplotlib.pyplot as plt

import pandas as pd



#unione_features = []

features_per_muscolo = {}

#LEGGO IL DATAFRAME(esempio)
#path = "C:/Users/omobo/Desktop/DataframeNormalizzati/Dataframe_VL.csv"
path_1 = "C:/Users/omobo/Desktop/DataframeVM/Dataframe_VM_label.csv"
label = os.path.splitext(path_1)[0].split('_')[1]
df_1 = pd.read_csv(path_1)


path_2 = "C:/Users/omobo/Desktop/DataframeVM/Dataframe_VM_output.csv"
label = os.path.splitext(path_1)[0].split('_')[1]
df_2 = pd.read_csv(path_2)


#LISTA DELLE FEATURES COMPLETA
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

# for i in range(len(feature_names)):
#     print('---------------------------------------------------------------')
#     print(feature_names[i])
#     print(df.groupby('Class')[feature_names[i]].describe())



 
#SEPARO IL DataFrame INIZIALE IN 2 DF: UNO PER FSHD E UNO PER MD1
  
MD1 = df_1  #solo per non cambiare tutto il codice
FSHD = df_2

#PLOT DI TUTTE LE DISTRIBUZIONI 

for feature in feature_names:
    f, axarr = plt.subplots(1,3,figsize=(20,5));
    n, bins, patches = axarr[0].hist(MD1[feature], bins= 10,  alpha=0.75)
    axarr[0].set_title("GROUND_TRUTH_{}".format(feature))
    axarr[0].set_xlabel('a.u.')  #arbitrary unit 
    axarr[0].set_ylabel('frequency')   #frequenza di ripetiione di quel dato  
    
    n, bins, patches = axarr[1].hist(FSHD[feature], bins=10, fc = 'orange',ec = 'orange', alpha=0.9)
    axarr[1].set_title("UNET_3D_PRED_{}".format(feature))
    axarr[1].set_xlabel('a.u.')  #arbitrary unit
    axarr[1].set_ylabel('frequency')   #frequenza di ripetiione di quel dato    
    
    ###Label Histogram
    n, bins, patches = axarr[2].hist(MD1[feature], bins=10, alpha=0.7 ,label='GT')
    n, bins, patches = axarr[2].hist(FSHD[feature], bins=10,  alpha=0.7,label='PRED')
    axarr[2].set_title('GT vs PRED')
    axarr[2].set_xlabel('a.u')
    axarr[2].set_ylabel('frequency')
    axarr[2].legend(loc='upper right')

         
    f.subplots_adjust(wspace=0.3, hspace=0, top=0.8)
    plt.show();
    
#LE FEATURE ESTRATTE CON LE PRES DONO UGUALI O DIVERSE A QUELLE ESTRATTE CON LE SEGM MANUALI?
#SE ANCHE CON IL CASO PEGGIORE (PRED UNET) SONO UGUALI ALLORA SI POSSONO USARE
#ALTRIMENTI VEDIAMO COL CASO MIGLIORE

    
#DISTRIBUZIONI NORMALI? --> test shapiro wilk

feature_normali = []
for i in range(len(feature_names)):
    print('---------------------------------------------------------------')
    print(feature_names[i])
    print('MD1')
    print(stats.shapiro(MD1[feature_names[i]]))
    statistic,pvalue_md1 = stats.shapiro(MD1[feature_names[i]])
    print('FSHD')
    print(stats.shapiro(FSHD[feature_names[i]]))
    statistic,pvalue_fshd = stats.shapiro(FSHD[feature_names[i]])
    if pvalue_md1 > 0.05 and pvalue_fshd > 0.05:     #vuol dire che sono distribuzioni normali [https://pythonfordatascienceorg.wordpress.com/independent-t-test-python/]
        feature_normali.append(feature_names[i])
        
               


#procediamo con il t-test 
#quali features sono significative per discriminare
#il vasto intermedio in un soggetto con MD1 da quello con FSHD?

#dopo aver selezionato le features normali , applico il test di levene per verificare l'omogeneità della varianza

feature_UGUALI = []
for i in range(len(feature_normali)):
    print('---------------------------------------------------------------')
    print(feature_normali[i])
    st,pv = stats.levene(MD1[feature_normali[i]], FSHD[feature_normali[i]])  #la varianza è omogenea? sì--> t test, no--> t-test di Wilk's (equal_var = False)
    if pv > 0.05:  #il test non è significativo significa che c'è omogeneità
        print(stats.ttest_ind(MD1[feature_normali[i]], FSHD[feature_normali[i]])) 
        statistics,pvalue = stats.ttest_ind(MD1[feature_normali[i]], FSHD[feature_normali[i]])
    else:
        print(stats.ttest_ind(MD1[feature_normali[i]], FSHD[feature_normali[i]],equal_var= False))
        statistics,pvalue = stats.ttest_ind(MD1[feature_normali[i]], FSHD[feature_normali[i]],equal_var= False)
    
   
    
    if pvalue > 0.05:  #--> features che sono "uguali" sia calcolate dalla segm che dalla pred
      feature_UGUALI.append(feature_normali[i])
      
      
features_per_muscolo[label] = feature_UGUALI
   



feature_name_interesting = []
for i in range(len(feature_names)):
    print('---------------------------------------------------------------')
    print(feature_names[i])
    st,pvalue = stats.mannwhitneyu(MD1[feature_names[i]], FSHD[feature_names[i]])
    print(stats.mannwhitneyu(MD1[feature_names[i]], FSHD[feature_names[i]]))
    if pvalue > 0.05: 
      feature_name_interesting.append(feature_names[i])
      # if features_normali[i] not in unione_features:
      #     unione_features.append(features_normali[i])

features_per_muscolo[label] = feature_name_interesting