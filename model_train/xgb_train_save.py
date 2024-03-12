# -*- coding: utf-8 -*-
"""
Python script to train a xgb model for downscaling satellite observations of NO2 to near-surface concentration maps

Reference: 
Minsu Kim, Dominik Brunner, Gerrit Kuhlmann (2021) 
Importance of satellite observations for high-resolution mapping of near-surface NO2 by machine learning, 
Remote sensing of Environment DOI: https://doi.org/10.1016/j.rse.2021.112573

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn.metrics as metrics
import pickle
import os
import argparse
import textwrap
from xgboost import XGBRegressor

# %% functions
def get_bbox(ROI):
    """
    making a bounding box of the region of interest (ROI)
    """
    if ROI == 'ROI1': #alpine region
        bbox = [6,12,42,48]
    else:# Benelux region
        bbox = [2,8,48,54]        
    return bbox    

def get_mask(df,bbox):
    """
    masking the dataframe based on the bounding box (ROI)
    """
    return (df.Longitude>bbox[0])&(df.Longitude<bbox[1])&(df.Latitude>bbox[2])&(df.Latitude<bbox[3])

def make_dummy_all(df):
    """
    Using the metadata of land use :: make dummies out of categorical information of land use
    """
    filename =  os.path.join(root, 'input','LUD','clc_legend.csv')
    metaLUD = pd.read_csv(filename, index_col=False, sep=';')
    possibilites = metaLUD.CLC_CODE.unique()
    exists = df.lu.unique()
    difference = pd.Series([item for item in possibilites if item not in exists])
    target = df.lu.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    dummy = pd.get_dummies(target)
    dummy = dummy.drop(dummy.index[list(range(len(dummy)-len(difference), len(dummy)))])
    targetMajor = np.floor(target/100.)
    dummyM = pd.get_dummies(targetMajor, prefix='lu_major')
    dummyM = dummyM.drop(dummyM.index[list(range(len(dummy)-len(difference), len(dummy)))])

    for hname in list(dummy):
        namela = metaLUD.LABEL3.loc[metaLUD.CLC_CODE==hname].to_string(index=False)
        namela = namela.strip().replace(' ', '_')
        dummy = dummy.rename(columns={hname:namela})
    
    return dummy, dummyM          

def make_dummy_station_type(df):
    """
    make dummies out of categorical station types
    """
    df[df=='industry']='industrial'
    possibilites = ['background', 'traffic', 'industrial']
    exists = df.unique()
    difference = pd.Series([item for item in possibilites if item not in exists])
    target = df.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    if len(difference) == 0:
       dummy = pd.get_dummies(target)
    else:       
       dummy = pd.get_dummies(target)
       dummy = dummy.drop(dummy.index[list(range(len(dummy)-len(difference), len(dummy)))])    
    return dummy[possibilites]          

def make_dummy_station_area(df):
    """
    make dummies out of categorical station area
    """    
    possibilites = ['rural', 'urban', 'suburban', 'rural-regional', 'rural-remote','rural-nearcity', 'highmountain']
    exists = df.unique()
    difference = pd.Series([item for item in possibilites if item not in exists])
    target = df.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    if len(difference) == 0:
       dummy = pd.get_dummies(target)
    else:       
       dummy = pd.get_dummies(target)
       dummy = dummy.drop(dummy.index[list(range(len(dummy)-len(difference), len(dummy)))])
    return dummy[possibilites]     
        
def make_dummy_smod(df):
    """
    make dummies out of categorical smod (degree of urbanisation)
    """    
    possibilites = ['smod_rural', 'smod_urban', 'smod_suburban','smod_none']
    exists = df.unique()
    difference = pd.Series([item for item in possibilites if item not in exists])
    target = df.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    if len(difference) == 0:
       dummy = pd.get_dummies(target)
    else:       
       dummy = pd.get_dummies(target)
       dummy = dummy.drop(dummy.index[list(range(len(dummy)-len(difference), len(dummy)))])
          
    return dummy[possibilites]      

def make_dummy_class(df, ncluster):
    """
    make dummies out of categorical classes (needed for stratified sampling - validation/test set)
    """
    possibilites = []
    for nid in range(int(ncluster)):
        possibilites.append(np.str(nid)+'.0')
    exists = []
    for item in df.unique():
        exists.append(np.str(item))

    difference = pd.Series([item for item in possibilites if item not in exists])
    target = df.append(pd.Series(difference))
    target = target.reset_index(drop=True)
    if len(difference) == 0:
       dummy = pd.get_dummies(target,prefix='class_id')
    else:       
       dummy = pd.get_dummies(target,prefix='class_id')
       dummy = dummy.drop(dummy.index[list(range(len(dummy)-len(difference), len(dummy)))])
    
    listid = []
    for poss_id in possibilites:
        listid.append('class_id_'+poss_id)
        
    return dummy[listid]                  
  

def data_transform_preprocess_omi_filled(df, model_ID, qa=50):
    """
    make dataframe of co-variates for training the model (here, we use the gap-filled data of TROPOMI) 
    """
    df.loc[df.Concentration<=0,'Concentration'] = np.nan
    df.loc[df.Concentration>df.Concentration.quantile(q=0.99999),'Concentration'] = np.nan
    df.loc[df.Concentration<df.Concentration.quantile(q=0.00001),'Concentration'] = np.nan
    trafficmask = np.isnan(df.rld)
    trafficmask2 = (df.tfv<0)
    df.loc[trafficmask2,'tfv'] = np.nan    
    df.loc[trafficmask,'tfv'] = 0
    df.loc[trafficmask,'rld'] = 0
    df.loc[np.isnan(df.lu),'lu'] = 523 # assgin sea and oceans    

    df1 = pd.DataFrame()  
    # location of stations
    df1['Longitude'] = df.lon
    df1['Latitude'] = df.lat
    # no2 observations
    if qa==75:
        df1['Certainty_dist'] = df.hourly_certainty_dist_qa75
    else: 
        df1['Certainty_dist'] = df.hourly_certainty_dist
       
    # meteological data
    df1['10m_u-component_of_wind_speed'] = df.u10
    df1['10m_v-component_of_wind_speed'] = df.v10
    df1['10m_wind_speed'] = np.linalg.norm([df.u10,df.v10], axis=0)
    df1['Temperature_2m'] = df.t2m
    df1['Solar_radiation'] = df.cdir
    df1['Total_precipitation'] = df.tp    
    df1['Boundary_layer_height'] = df.blh    
    # elevation features
    df1['Digital_Elevation_Map'] = df.dem
    df1['Digital_Elevation_Map_wt_400m'] = df.dem_hpf_400m
    df1['Digital_Elevation_Map_wt_1km'] = df.dem_hpf_1km
    df1['Digital_Elevation_Map_wt_6km'] = df.dem_hpf_6km
    df1['Digital_Elevation_Map_wt_25km'] = df.dem_hpf_25km
    df1['Digital_Elevation_Map_wt_100km'] = df.dem_hpf_100km

    # emission source realated features    
    df1['Population_density_f'] = np.log1p(df.popf_max) #to keep the information of zero values, shift by 1
    df1['Population_density_d'] = df.popf_dist #to keep the information of zero values, shift by 1    
    df1['Traffic_volume_f'] = np.log1p(df.tfvf_max)
    df1['Traffic_volume_d'] = df.tfvf_dist    
    df1['Road_length_density_f'] = np.log1p(df.rldf_max)
    df1['Road_length_density_d'] = df.rldf_dist
    df1['TNO_emission_f'] = np.log1p(df.tnof_max)
    df1['TNO_emission_d'] = df.tnof_dist
    df1['Land use'] = df.lu
    df1['Population_density'] = np.log1p(df['pop']) #to keep the information of zero values, shift by 1
    df1['Traffic_volume'] = np.log1p(df.tfv)
    df1['Road_length_density'] = np.log1p(df.rld)    
    # adding time info as features
    df1['Month'] = df.index.month
    df1['Day'] = df.index.day
    df1['Hour'] = df.index.hour
    df1['weekday'] = 1*(df.index.dayofweek<5)+ 2*(df.index.dayofweek==5) + 3*(df.index.dayofweek==6)
    df1['doy'] = df.index.dayofyear
    
    # co-variates are min-max scaled and the scaler is saved for prediction
    df1 = df1.replace([np.inf, -np.inf], np.nan)
    min_max_scaler = preprocessing.MinMaxScaler()    
    scale_data = min_max_scaler.fit(df1)    
    df_train_minmax = min_max_scaler.fit_transform(df1)
    df1 = pd.DataFrame(df_train_minmax, columns=list(df1))
    df1 = df1.set_index(df.index)   
    pickle.dump(scale_data, open(os.path.join(root, 'models', model_ID, 'min_max_scaler_variables.sav'), 'wb'))    
  
    # NO2 concentrations (satelite, near-surface) are power-transformed and the scaler is saved for prediction
    pw_scaler = preprocessing.PowerTransformer()    
    if qa==75:
        scale_no2 = pw_scaler.fit(df.hourly_no2_avg_filled_qa75.values.reshape(-1,1))
        df1['Sentinel5p_no2_fill'] = pw_scaler.fit_transform(df.hourly_no2_avg_filled_qa75.values.reshape(-1,1))   
        pickle.dump(scale_no2, open(os.path.join(root, 'models',model_ID, 'pw_scaler_filled_sat.sav'), 'wb'))  
    else:
        scale_no2 = pw_scaler.fit(df.hourly_no2_avg_filled.values.reshape(-1,1))
        df1['Sentinel5p_no2_fill'] = pw_scaler.fit_transform(df.hourly_no2_avg_filled.values.reshape(-1,1))   
        pickle.dump(scale_no2, open(os.path.join(root, 'models',model_ID, 'pw_scaler_filled_sat.sav'), 'wb'))  

    scale_no2 = pw_scaler.fit(df.Concentration.values.reshape(-1,1))
    df1['Airbase_Concentration'] = pw_scaler.fit_transform(df.Concentration.values.reshape(-1,1))   
    pickle.dump(scale_no2, open(os.path.join(root, 'models',model_ID, 'pw_scaler_0.001percent.sav'), 'wb'))        
                           
    dummy, dummyM = make_dummy_all(df)
    dummy = dummy.set_index(df.index)
    dummyM = dummyM.set_index(df.index)
          
    df_train = df1    
    df_train['AirQualityStationEoICode'] = df['AirQualityStationEoICode'] 
    df_train['AirQualityStationArea'] = df['AirQualityStationArea'] 
    df_train['AirQualityStationType'] = df['AirQualityStationType'] 
    df_train = pd.concat([df_train, dummy], axis=1) 
    
    return df_train.dropna()

def model_sort_features(df_train,modeln = 'Model0'):
    """
    Here, differnt models indicate the omission/inclusion of different co-variates for comparisons
    """
    switcher = {
        "Model1": df_train.drop(columns=['Population_density_f', 'Population_density_d', 'Traffic_volume_f', 'Traffic_volume_d', 'Road_length_density_f', 'Road_length_density_d',
                                         'TNO_emission_f', 'TNO_emission_d', 'Month', 'Day','10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed','Land use']),
        "Model2": df_train.drop(columns=['Digital_Elevation_Map_wt_400m','Digital_Elevation_Map_wt_1km','Digital_Elevation_Map_wt_6km', 'Digital_Elevation_Map_wt_25km', 'Digital_Elevation_Map_wt_100km', # no elevation features
                                         'Population_density', 'Traffic_volume', 'Road_length_density', 'Month','Day','10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed','Land use']),
        "Model3": df_train.drop(columns=['Population_density_f', 'Population_density_d', 'Traffic_volume_f', 'Traffic_volume_d', 'Road_length_density_f', 'Road_length_density_d',
                                         'TNO_emission_f', 'TNO_emission_d', '10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed','Land use',
                                         'Digital_Elevation_Map_wt_400m','Digital_Elevation_Map_wt_1km','Digital_Elevation_Map_wt_6km', 'Digital_Elevation_Map_wt_25km', 'Digital_Elevation_Map_wt_100km',                                        
                                         'Month','Day']),  # when the time features are not included?
        "Model4": df_train.drop(columns=['Population_density', 'Traffic_volume', 'Road_length_density', 'doy']),
    }
    return switcher.get(modeln, "Provide new name")


def umap_clustering(df_train):
    """
    For stratified sampling, AQ stations are classified after clustering (using umap-mapping then hdbscan for grouping) 
    Here co-variatees included in the model are used for classification
    when cvID is given, pre-saved calssification can be called.
    """
    filename = 'umap_classification_stns_index_%d.pkl' % (cvID) 
    if not os.path.isfile(os.path.join(root,'models',model_ID, filename)): 
        import umap.umap_ as umap
        from sklearn.decomposition import PCA
        import hdbscan
        temp = df_train.groupby(df_train.AirQualityStationEoICode).mean()
        X = temp.drop(columns={'Sentinel5p_no2_fill', 'Certainty_dist', 'Hour', 'weekday', 'doy', 'Airbase_Concentration','Longitude', 'Latitude'})
        
        cluster_count_std = 50
        while (cluster_count_std > 30):
            lowd_df = PCA(n_components = 5).fit_transform(X)
            embedding = umap.UMAP(n_neighbors=5, metric='correlation', min_dist=0.1,local_connectivity=3).fit_transform(lowd_df)
            hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=30).fit_predict(embedding)       
            dfclass = pd.DataFrame()
            dfclass['umap_1'] = embedding[:,0]
            dfclass['umap_2'] = embedding[:,1]
            dfclass['hdbscan'] = hdbscan_labels
            dfclass['hdbscan'] ='class_'+dfclass['hdbscan'].apply(str)
            dfclass.hdbscan[dfclass['hdbscan'] == 'class_-1'] = str('not clustered')
            cluster_count_std = dfclass.groupby('hdbscan').count().std().umap_1
        
        X = X.reset_index()
        X = pd.concat([X, dfclass], axis=1) 
        X.to_pickle(os.path.join(root,'models',model_ID, filename))   
    else:
        print('Stations are clustered and saved as cvID of %d: loading the saved file'%cvID)  
        X = pd.read_pickle(os.path.join(root,'models',model_ID, filename))   
    return X

def split_train_test_umap_clustering_fixed_test(df_train, train_size, root, model_ID, idn=0):
    """
    For stratified sampling, the umap-clustered staions are divided to a train set and a test set.
    Here 'cvID' is an index for clustering with umap 
    Once umap-clustering is performed, it can be saved and called later use. 
    When it is desired, the test stations can be reselected within the clusters of index cvID.
    the selection of test stations can be saved with other index, idn
    """
    filename = 'train_stns_fixed_test_'+'_train_size_%.1f_index_%d_%d.pkl' % (train_size, cvID, idn)     
    
    if not os.path.isfile(os.path.join(root,'models',model_ID, filename)):  
        fname = 'umap_classification_stns_index_%d.pkl' % (cvID) 
        if not os.path.isfile(os.path.join(root,'models',model_ID, fname)): 
            X_old = umap_clustering(df_train)
        else:
            X_old = pd.read_pickle(os.path.join(root,'models',model_ID, fname))
            
        fname = 'train_stns_umap_train_size_0.9_index_%d_%d.pkl' % (cvID, idn) 
        train_stns_old = pd.read_pickle(os.path.join(root,'models',model_ID, fname)) 
        X = X_old[X_old['AirQualityStationEoICode'].isin(train_stns_old)].reset_index(drop=True)           
        test_stns = X_old['AirQualityStationEoICode'][~X_old['AirQualityStationEoICode'].isin(train_stns_old)].reset_index(drop=True)
        
        cluster_ids = X.hdbscan.unique()
        train_stns = []
        for c_id in cluster_ids:
            nstns = len(X.loc[X['hdbscan']==c_id])
            n_train_stns = int(nstns*train_size)
            train_stns.append(X['AirQualityStationEoICode'].iloc[np.random.choice(X.loc[X['hdbscan']==c_id].index,n_train_stns,replace=False)])      
               
        train_stns = pd.concat(train_stns).sample(frac=1).reset_index(drop=True) 
        train_stns.to_pickle(os.path.join(root,'models',model_ID, filename))           
    else:
        fname = 'umap_classification_stns_index_%d.pkl' % (cvID) 
        X_old = pd.read_pickle(os.path.join(root,'models',model_ID, fname))       
        fname = 'train_stns_umap_train_size_0.9_index_%d_%d.pkl' % (cvID, idn) 
        train_stns_old = pd.read_pickle(os.path.join(root,'models',model_ID, fname)) 
        test_stns = X_old['AirQualityStationEoICode'][~X_old['AirQualityStationEoICode'].isin(train_stns_old)].reset_index(drop=True)       
        train_stns = pd.read_pickle(os.path.join(root,'models',model_ID, filename))     
                       
    X_train, y_train = separate_target_features(df_train[df_train['AirQualityStationEoICode'].isin(train_stns)].sample(frac=1))
    X_test, y_test = separate_target_features(df_train[~df_train['AirQualityStationEoICode'].isin(train_stns)].sample(frac=1))
    X_valid, y_valid = separate_target_features(df_train[df_train['AirQualityStationEoICode'].isin(test_stns)].sample(frac=1))

    return X_train, X_test, y_train, y_test, X_valid, y_valid

def inverse_scale(prediction,model_ID):
    """
    Invserse scaler of porwer-scaled values of NO2
    """  
    scalers = pickle.load(open(os.path.join(root, 'models',model_ID,'pw_scaler_0.001percent.sav'), 'rb'))       
    return scalers.inverse_transform(prediction)

def get_r2_on_real_values(exported_pipeline,X,y):
    """
    This caclulates the r2 of the model prediction on the real values (the model predicts in porwer-scaled values)
    """   
    y_pred_real = inverse_scale(exported_pipeline.predict(X).reshape(-1,1),model_ID)
    y_obs_real = inverse_scale(y.values.reshape(-1,1),model_ID)
    return metrics.r2_score(y_obs_real, y_pred_real)

def separate_target_features(df_train):
    """
    Prepare d-matrix for model training (removing other information that are not used in the model)
    """   
    y = df_train.Airbase_Concentration
    # remove categorical values
    X = df_train.drop(columns=['Airbase_Concentration','AirQualityStationArea',
                               'AirQualityStationEoICode','AirQualityStationType',
                               'Longitude', 'Latitude'])
    return X, y


def train_save_XGB_models_fixed_test_set(df_train, train_size):
    """
    train and save the XGB model
    """   
    X, y = separate_target_features(df_train)
    X_train, X_test, y_train, y_test, X_valid, y_valid = split_train_test_umap_clustering_fixed_test(df_train, train_size, root, model_ID)    

    exported_pipeline = XGBRegressor(tree_method='gpu_hist', 
                                     learning_rate=0.1, 
                                     max_depth=15, 
                                     min_child_weight=12, 
                                     n_estimators=100,
                                     n_jobs=12, 
                                     nthread=1, 
                                     subsample=0.9500000000000001) # These hyperparameters are optimised using Optuna
    
    exported_pipeline.fit(X_train, y_train)    
    accuracy_dt_test = get_r2_on_real_values(exported_pipeline,X_test,y_test)
    accuracy_dt_train = get_r2_on_real_values(exported_pipeline,X_train,y_train)
    accuracy_dt_all = get_r2_on_real_values(exported_pipeline,X,y)   
    accuracy_dt_valid = get_r2_on_real_values(exported_pipeline,X_valid,y_valid)    

    filename = 'XGB_fixed_test_'+ modeln+'_train_size_%.1f_%.4f_%.4f_%.4f_%.4f_index_%d.sav' % (train_size, accuracy_dt_test, accuracy_dt_train, accuracy_dt_all,accuracy_dt_valid,cvID) 
    print(filename)
    pickle.dump(exported_pipeline, open(os.path.join(root,'models',model_ID,filename), 'wb'))
    

def main():
    
    description = textwrap.dedent("""\   
        This script is to train XGB models and to calculate cross-validation score for different different train/validation set that are selected based on umap clustering
        Here, there are several models using different features
        ---------------------------------------        
        Model1: Null model : Using data without extraction of particular features as other models
        Model2. Spatial features : Feature extraction of emission sources (max values and max distance in range [0,100])
        Model3. Spatial features + wavelet transform of DEM (digital elevation model) 
        Model4. dimensionality reduction  (rebate), remove some features 
        -------------------------------------
        by Minsu Kim (minsu.kim@empa.ch)
        ORCID:https://orcid.org/0000-0002-3942-3743
    """)
        
    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('train_size', type=float, help='size of train set proportional to the all data (number between 0 and 1)')
    parser.add_argument('Model', type=str, help='model name (Model1, Model2, Model3, Model4)')
    parser.add_argument('qa', type=int, help='where to cut the training data; qualitiy assurance level of 50 or75')
    args = parser.parse_args()

    global root,modeln, model_ID, cvID

    modeln = args.Model
    train_size = args.train_size
    qa_val = args.qa
    
    root = '.' # PATH 
    model_ID = 'Full_features_'+str(qa_val)     
    cvID = 0 #This index is to indicate the clustered AQ stations using UMAP 

    # Here we use already prepared data called 'ROI1_full_features_qa75.pkl' for every stations
    # Then the data will be trasformed to be fed to train a xgb model using the function 'data_transform_preprocess_omi_filled'
    # The data transformed/cleaned data is saved as df_data.csv and df_data.pkl, which can be used later for training another model
    if not os.path.isfile(os.path.join(root,'models',model_ID, 'df_data.pkl')):
        df = pd.read_pickle(os.path.join(root, 'data','ROI1_full_features_qa75.pkl'))
        df_train = data_transform_preprocess_omi_filled(df,model_ID, qa=qa_val)
        if not os.path.isdir(os.path.join(root,'models',model_ID)):
            os.mkdir(os.path.join(root,'models',model_ID)) 
        df_train.to_csv(os.path.join(root,'models',model_ID, 'df_data.csv'))  
        df_train.to_pickle(os.path.join(root,'models',model_ID, 'df_data.pkl'))   
    else:
        df_train = pd.read_pickle(os.path.join(root,'models',model_ID, 'df_data.pkl'))   

    # select co-variates (features) that will be used in training
    df_train_agg = model_sort_features(df_train,modeln)    
    
    # The data may be collected for an extended period, here select the study period
    df_train_agg = df_train_agg.loc[~((df_train_agg.index.year==2018)&(df_train_agg.index.month<6))]
    df_train_agg = df_train_agg.loc[~((df_train_agg.index.year==2020)&(df_train_agg.index.month>6))]

    train_save_XGB_models_fixed_test_set(df_train_agg, train_size)
      
if __name__ == '__main__':
    main()

