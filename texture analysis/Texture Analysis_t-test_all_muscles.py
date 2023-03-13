# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:26:34 2023

@author: omobo
"""
import os
from glob import glob
import numpy as np
from scipy import stats 

import pandas as pd


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import plotly.express as px



features_per_muscolo ={}
#ho un dataframe
features_normali_per_muscolo ={}


unione_features = []
#leggo il dataframe

#ITERATIVAMENTE, APPLICO IL T-TEST ED ESTRAGGO LE FEATURES DA TUTTI I DATAFRAME
directory = "C:/Users/omobo/Desktop/Dataframe"

df_paths = sorted(glob(os.path.join(directory,'*')))

for path in df_paths:
    label = os.path.splitext(path)[0].split('_')[1]
    df = pd.read_csv(path)

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

    for i in range(len(feature_names)):
        print('---------------------------------------------------------------')
        print(feature_names[i])
        print(df.groupby('Class')[feature_names[i]].describe())
        
    MD1 = df[(df['Class'] == 'MD1')]
    FSHD = df[(df['Class'] == 'FSHD')]  
    
        
    #hanno una distribuzione normale?
    
    
    features_normali = []
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
            features_normali.append(feature_names[i])
            
    features_normali_per_muscolo[label]=features_normali   
                   


    #procediamo con il t-test vero e proprio
    #quali features sono significative per discriminare
    #il vasto intermedio in un soggetto con MD1 da quello con FSHD?

          
    feature_name_interesting = []
    for i in range(len(features_normali)):
        print('---------------------------------------------------------------')
        print(features_normali[i])
        st,pv = stats.levene(MD1[features_normali[i]], FSHD[features_normali[i]])  #la varianza è omogenea? sì--> t test, no--> t-test di Wilk's (equal_var = False)
        if pv > 0.05:  #il test non è significativo significa che c'è omogeneità
            print(stats.ttest_ind(MD1[features_normali[i]], FSHD[features_normali[i]])) 
            statistics,pvalue = stats.ttest_ind(MD1[features_normali[i]], FSHD[features_normali[i]])
        else:
            print(stats.ttest_ind(MD1[features_normali[i]], FSHD[features_normali[i]],equal_var= False))
            statistics,pvalue = stats.ttest_ind(MD1[features_normali[i]], FSHD[features_normali[i]],equal_var= False)
        
        if pvalue < 0.05:  #--> differenza significativa!
          feature_name_interesting.append(features_normali[i])
          if features_normali[i] not in unione_features:
              unione_features.append(features_normali[i])
    
    features_per_muscolo[label] = feature_name_interesting
       

   
#%%ora voglio solo estrarre le features interessanti. distribuzione normale e differenza statisticamente significativa
#leggo i dataframes completi ( tutti i soggetti tutte le features : 107x89 per coscia)
#estraggo solo le colonne corrispondenti alle features selezionate PER QUEL MUSCOLO
      #faccio la pca solo con quelle
      
pca_results = {}

for path in df_paths:
    label = os.path.splitext(path)[0].split('_')[1]
    df = pd.read_csv(path)
    
    df_ridotto = df[features_per_muscolo[label]]
    
    data_subset = df_ridotto.values  
    data_subset_scaled = MinMaxScaler().fit_transform(data_subset)   #normalizzo le variabili perchè PCA risente della scala. 
  
    pca = PCA(n_components = 2)
    pca_results_2d = pca.fit_transform(data_subset_scaled)   #ottengo 3 principal components
    pca_results[label] = pca_results_2d  
    
#%% elimino gli outliers    
zoom_in_percentile_range = (0, 99)
df_plot_list = []
for path in df_paths:
    
    label = os.path.splitext(path)[0].split('_')[1]
    df = pd.read_csv(path)#df mi interessa solo per la colonna delle classi
     
    X = pca_results[label]
    cutoffs_0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_1 = np.percentile(X[:, 1], zoom_in_percentile_range)
    #cutoffs_2 = np.percentile(X[:, 2], zoom_in_percentile_range)


    no_outliers_mask = np.all(X > [cutoffs_0[0], cutoffs_1[0]], axis=1) & np.all(
            X < [cutoffs_0[1], cutoffs_1[1]], axis=1)

    pca_results_no_outliers = X[no_outliers_mask]

    #creo un dataframe con pca_results_no_outliers e le classi corrispondenti

    df_plot = pd.DataFrame(pca_results_no_outliers,columns=['pca_1','pca_2'])
    df_plot['Class'] = (df['Class'].values)[no_outliers_mask]
    df_plot_list.append(df_plot)
    #df plot segue lo stesso ordine di df_path quindi parte dal ADD
    
#%%visualizzazione    
    
for df_plot,path in zip(df_plot_list,df_paths):
    label = os.path.splitext(path)[0].split('_')[1]
    fig = px.scatter(
                df_plot, x= 'pca_1', y= 'pca_2',
                color= 'Class', labels={'color': 'Class'})
    fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01))
    fig.update_traces(marker_size=8)
    fig.update_layout(title= label, width=1400, height=500)

    fig.show()