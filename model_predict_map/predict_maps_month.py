# -*- coding: utf-8 -*-
"""
Python script to generate a map of near-surface NO2 concentration using a trained XGB model
For the outcome, all co-variates should be prepared in maps (2d arrays)

Reference: 
Minsu Kim, Dominik Brunner, Gerrit Kuhlmann (2021) 
Importance of satellite observations for high-resolution mapping of near-surface NO2 by machine learning, 
Remote sensing of Environment DOI: https://doi.org/10.1016/j.rse.2021.112573

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""
# %% import packages

import xarray as xr
import pandas as pd
import glob
import numpy as np
import os
import pickle
import argparse
import textwrap
from datetime import datetime, timedelta

# %% functions
def parse_date(s):
    if len(s) == 10:
        return datetime.strptime(s, '%Y-%m-%d')
    else:
        return datetime.strptime(s, '%Y-%j')

def first_hours_from_last_month(root, yeari, monthi):
    firstday = str(yeari)+'-'+str(monthi).zfill(2)+'-01' 
    date = parse_date(firstday) + timedelta(days=-1)
    ds = xr.open_dataset(os.path.join(root, 'data','Sentinelno2','omi_filled','S5P_hourly_NO2_filled_'+date.strftime('%Y')+'-'+date.strftime('%m')+'-'+date.strftime('%d')+'.nc'))
    return ds

def read_ERA5_time_series_month_data(varname, yeari, monthi, root, SSA):
	filenames = 'meteo_interp_' +varname+'_'+str(yeari)+str(monthi).zfill(2)+'*.nc'
	all_files = glob.glob(os.path.join(root, 'data','ERA5', ROI, filenames))
	all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
	ds_concat = xr.concat(all_dsets, dim='time')
	return ds_concat.sel(**SSA)

def read_NO2_hourly_omi_filled_time_series_month_data(root, yeari, monthi,SSA):
    all_files = sorted(glob.glob(os.path.join(root, 'data','Sentinelno2','omi_filled','S5P_hourly_NO2_filled_'+str(yeari)+'-'+str(monthi).zfill(2)+'-*.nc')))
    all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
    additional_ds = first_hours_from_last_month(root, yeari, monthi)
    all_dsets.append(additional_ds)
    ds_concat = xr.concat(all_dsets, dim='time')
    return ds_concat.sel(**SSA)

def make_dummy_all(df,root):
    # metadata of land use :: make dummies out of categorical information of land use
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

def data_transform_preprocess_omi_filled(df, model_ID):

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
    #df1['Sentinel5p_no2_fill'] = df.hourly_no2_avg_filled
    df1['Certainty_dist'] = df.certainty_dist
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
    
    df1 = df1.replace([np.inf, -np.inf], np.nan)
    scalers = pickle.load(open(os.path.join(root, 'models',model_ID, 'min_max_scaler_variables.sav'), 'rb'))
    df_train_minmax = scalers.transform(df1)
    df1 = pd.DataFrame(df_train_minmax, columns=list(df1))
    df1 = df1.set_index(df.index)   
    
    scalers = pickle.load(open(os.path.join(root, 'models',model_ID,'pw_scaler_filled_sat.sav'), 'rb'))
    df1['Sentinel5p_no2_fill'] = scalers.transform(df.no2_avg_filled.values.reshape(-1,1))
     
    dummy, dummyM = make_dummy_all(df,root)
    dummy = dummy.set_index(df.index)
    dummyM = dummyM.set_index(df.index)
          
    df_train = pd.concat([df1, dummy], axis=1) 
    
    return df_train.dropna()

def model_sort_features(df_train,modeln = 'Model0'): 
    switcher = { 
        "Model1": df_train.drop(columns=['Digital_Elevation_Map_wt_400m','Digital_Elevation_Map_wt_1km','Digital_Elevation_Map_wt_6km', 'Digital_Elevation_Map_wt_25km', 'Digital_Elevation_Map_wt_100km',# no elevation features
                                         'Population_density_f', 'Population_density_d', 'Traffic_volume_f', 'Traffic_volume_d', 'Road_length_density_f', 'Road_length_density_d',
                                         'TNO_emission_f', 'TNO_emission_d', 'doy']), 
        "Model2": df_train.drop(columns=['Digital_Elevation_Map_wt_400m','Digital_Elevation_Map_wt_1km','Digital_Elevation_Map_wt_6km', 'Digital_Elevation_Map_wt_25km', 'Digital_Elevation_Map_wt_100km', # no elevation features
                                         'Population_density', 'Traffic_volume', 'Road_length_density', 'doy']), 
        "Model3": df_train.drop(columns=['Population_density', 'Traffic_volume', 'Road_length_density', 'Month','Day','Hour', 'doy']),  # when the time features are not included?
        "Model4": df_train.drop(columns=['Population_density', 'Traffic_volume', 'Road_length_density', 'doy']), 
        "Model5": df_train.drop(columns=['Population_density', 'Traffic_volume', 'Road_length_density', 'Day','10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed']), 
        "Model0": df_train.drop(columns=['Population_density', 'Traffic_volume', 'Road_length_density', 'Day','Month','10m_u-component_of_wind_speed', '10m_v-component_of_wind_speed','Longitude', 'Latitude','Land use']),
    }      
    return switcher.get(modeln, "Provide new name") 


def sector_prediction(data_to_predict):
    def inverse_scale(prediction):
        scalers = pickle.load(open(os.path.join(root, 'models',model_ID, 'pw_scaler_0.001percent.sav'), 'rb'))       
        df_train_minmax = scalers.inverse_transform(prediction)
        return df_train_minmax 
    bstmodel=pickle.load(open(os.path.join(root, 'models', model_ID, 'XGB_best_'+modeln+'.sav'), 'rb'))    
    prediction = pd.DataFrame({'no2':bstmodel.predict(data_to_predict)}).set_index(data_to_predict.index)
    prediction = inverse_scale(prediction)    
    return pd.DataFrame({'no2':prediction.flatten()}, index=data_to_predict.index).to_xarray()   

def save_prediction(ds_series):
    t_obs = ds_series.time
    filename = ROI+'_'+modeln+r'_no2_'+str(t_obs.values)[:13]+'_lon_%5f_lat_%5f.nc' % (ds_series.lon[0].values, ds_series.lat[0].values)   
    target = os.path.join(root, 'models',model_ID, 'Results',filename)
    if not os.path.isfile(target): 
        df = ds_series.to_dataframe()
        X = data_transform_preprocess_omi_filled(df.reset_index().set_index('time'),model_ID)
        X = X[listf]
        X = X.reset_index().set_index(df.index)
        X = X.drop(columns=['time'])
        pred = sector_prediction(X)
        pred.coords['time'] = t_obs
        pred = pred.expand_dims('time')
        pred.to_netcdf(target) 
    else:
        print(filename)
        print('file exist')


# %%
def main():
    
    description = textwrap.dedent("""\
        make predictions of NO2 map with the best model using XGB (should be already saved and named as 'XGB_best_'+modeln+'.sav' )
        Here, there are several models (named 'modeln') using different features
        ---------------------------------------        
        Model1: Null model : Using data without extraction of particular features as other models
        Model2. Spatial features : Feature extraction of emission sources (max values and max distance in range [0,100])
        Model3. Spatial features + wavelet transform of DEM (digital elevation model) 
        Model4. dimensionality reduction  (rebate), remove some features 
        -------------------------------------
        
        This code requires 4 input arguments;
        arg1: Model name
        arg2: Year to predict (2018, 2019, 2020)
        arg3: Month to predict (1,2,3,4,5,6,7,8,9,10,11,12)
        arg4: qa threshold values for training set (50 or 75)
        
        As a result, there will subsections of predicted maps (netCDF files) of NO2 will be saved in the folder    
        os.path.join(root, 'models',model_ID, 'Results')
        with the name: ROI+'_'+modeln+r'_no2_'+str(t_obs.values)[:13]+'_lon_%5f_lat_%5f.nc' 
        36 netcdf files can be combiled to a single netcdf file for each time (hour)
        by using the script 'merge_predicted_ROI_maps.py'
    
        by Minsu Kim (minsu.kim@empa.ch)
        ORCID:https://orcid.org/0000-0002-3942-3743
        
    """)
        
    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    #parser.add_argument('ROI', type=str, help='region of interest (ROI1, ROI1, ALL)')
    parser.add_argument('Model', type=str, help='model name (Model1, Model2, Model3, Model4)')
    parser.add_argument('yeari', type=int, help='year to predict')
    parser.add_argument('monthi', type=int, help='month to predict')
    parser.add_argument('qa', type=int, help='where to cut the training data; qualitiy assurance level of 50 or75')

    global root, ROI,modeln,validationt, scoringm, model_ID, listf  
    
    args = parser.parse_args()
    modeln = args.Model
    yeari = args.yeari
    monthi = args.monthi
    qa_val = args.qa

    
    ROI = 'ROI1'
    root = '/scratch/snx3000/minsukim/'
    if ROI == 'ROI1':
        latlist = np.arange(42,48.5,1) #maps are generated by sections of 1 degree
        lonlist = np.arange(6,12.5,1) #maps are generated by sections of 1 degree
    else:
        latlist = np.arange(48,54.5,1)
        lonlist = np.arange(2,8.5,1)        
        
    model_ID = 'Full_features'+str(qa_val)   
    
    bstmodel=pickle.load(open(os.path.join(root, 'models', model_ID, 'XGB_best_'+modeln+'.sav'), 'rb'))    
    listf = bstmodel.get_booster().feature_names

    chunks = {'time':1,'lon':-1, 'lat':-1}

    for lonID in range(6):
        for latID in range(6):
            print(lonID)
            print(latID)
            SSA = dict(lat=slice(latlist[latID], latlist[latID+1]), lon=slice(lonlist[lonID], lonlist[lonID+1]))           
            tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
            featureData = xr.open_dataset(os.path.join(root, 'input', ROI+'_features_v2.nc'))
            demfeatureData = xr.open_dataset(os.path.join(root, 'input', ROI+'_dem_features_dmey.nc'))
            ds_inv = xr.merge([tinvData,featureData,demfeatureData], join ='outer').sel(**SSA)
            mfdsU10 = read_ERA5_time_series_month_data('u10', yeari, monthi, root, SSA)
            mfdsV10 = read_ERA5_time_series_month_data('v10', yeari, monthi, root, SSA)
            mfdsT2m = read_ERA5_time_series_month_data('t2m',yeari, monthi, root, SSA)
            mfdsCdir = read_ERA5_time_series_month_data('cdir',yeari, monthi, root, SSA)
            mfdsTp = read_ERA5_time_series_month_data('tp',yeari, monthi, root, SSA)
            mfdsBlh = read_ERA5_time_series_month_data('blh',yeari, monthi, root, SSA)
            mfdsMet = xr.merge([mfdsU10,mfdsV10,mfdsT2m,mfdsCdir,mfdsTp,mfdsBlh], join ='inner')
            _, index = np.unique(mfdsMet['time'], return_index=True)
            mfdsNo2 = read_NO2_hourly_omi_filled_time_series_month_data(root,yeari, monthi,SSA)
            mfdsNo2 = mfdsNo2.reindex_like(ds_inv,method='nearest')
            _, index = np.unique(mfdsNo2['time'], return_index=True)
            mfdsNo2 = mfdsNo2.isel(time=index)
            ds = xr.merge([mfdsMet,mfdsNo2,ds_inv], join ='inner')
            ds = ds.chunk(chunks=chunks)            
            ds = ds.transpose('time','lat','lon')
            ds = ds.sortby('time')
    
    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(save_prediction)(ds.isel(time=i)) for i in range(len(ds.time)))

if __name__ == '__main__':
    main()




