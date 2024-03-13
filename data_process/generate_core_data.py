# -*- coding: utf-8 -*-
"""
This script generates core dataset per stations 

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

# %% functions 

def find_stn_location(stnname):
    stnlon = metaAIRBASE.where(stnname,drop=True).Longitude[0]
    stnlat = metaAIRBASE.where(stnname,drop=True).Latitude[0]
    stnlon = lon.sel(lon=stnlon, method='nearest') #find nearest point for the selection of spatial domiain
    stnlat = lat.sel(lat=stnlat, method='nearest')
    return [np.asscalar(stnlat) , np.asscalar(stnlon)]

def read_ERA5_time_series_data(varname, root, SSA):
	filenames = 'meteo_interp_' +varname+'_*.nc'
	all_files = glob.glob(os.path.join(root, 'data','ERA5', ROI, filenames))
	all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
	ds_concat = xr.concat(all_dsets, dim='time')
	return ds_concat.sel(**SSA).to_dataframe()

def read_NO2_time_series_data(root, SSA):
	all_files = glob.glob(os.path.join(root, 'data','Sentinelno2',ROI,'S5P_NO2_*.nc'))
	all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
	ds_concat = xr.concat(all_dsets, dim='time')
	return ds_concat.sel(**SSA).to_dataframe()	

def save_data_from_near_stn(filename):    

    picklename = filename[-21:-4] +'_features.pkl'
    if not os.path.isfile(os.path.join(root, 'data','pickles', ROI, picklename)): 
        stnname = metaAIRBASE.AirQualityStationEoICode==filename[-21:-14] 
        try:
            stnloc = find_stn_location(stnname) 
            SSA = dict(lat=slice(stnloc[0]-regionrange, stnloc[0]+regionrange), lon=slice(stnloc[1]-regionrange, stnloc[1]+regionrange))
            mfdsU10 = read_ERA5_time_series_data('u10',root, SSA)
            mfdsV10 = read_ERA5_time_series_data('v10',root, SSA)
            mfdsT2m = read_ERA5_time_series_data('t2m',root, SSA)
            mfdsCdir = read_ERA5_time_series_data('cdir',root, SSA)
            mfdsTp = read_ERA5_time_series_data('tp',root, SSA)
            mfdsBlh = read_ERA5_time_series_data('blh',root, SSA)
            mfdsNo2 = read_NO2_time_series_data(root, SSA)
    
            met_stn_series = pd.merge(mfdsU10,mfdsV10,left_index=True,right_index=True,how='inner') #(default) merge only rows existing in df (left)
            met_stn_series = pd.merge(met_stn_series,mfdsT2m,left_index=True,right_index=True,how='inner') #(default) merge only rows existing in df (left)
            met_stn_series = pd.merge(met_stn_series,mfdsCdir,left_index=True,right_index=True,how='inner') #(default) merge only rows existing in df (left)
            met_stn_series = pd.merge(met_stn_series,mfdsTp,left_index=True,right_index=True,how='inner') #(default) merge only rows existing in df (left)
            met_stn_series = pd.merge(met_stn_series,mfdsBlh,left_index=True,right_index=True,how='inner') #(default) merge only rows existing in df (left)
    
            no2Timeseries = pd.read_csv(filename, usecols=[3,5])
            no2Timeseries['time'] = pd.to_datetime(no2Timeseries.DatetimeBegin, utc=True)
            no2Timeseries = no2Timeseries.set_index('time') #not strictly necessary to set index.. doing it because ds.to_dataframe() does it like that
            no2Timeseries.index = no2Timeseries.index.tz_convert(tz=None)
            no2Timeseries = no2Timeseries[no2Timeseries.index.year ==2018]

            df1 = pd.merge(met_stn_series,mfdsNo2,left_index=True,right_index=True,how='outer') #(default) merge only rows existing in df (left)
            df = pd.merge(no2Timeseries,df1,left_index=True,right_index=True,how='outer') # merge all rows and fill with nan
            df = df.drop(columns=['DatetimeBegin'])
    
            tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
            time_inv_input = tinvData.sel(**SSA).to_dataframe()
            for colname in list(time_inv_input):
                df[colname] = np.asscalar(time_inv_input[colname])     

            featureData = xr.open_dataset(os.path.join(root, 'input', ROI+'_features.nc'))
            feature_input = featureData.sel(**SSA).to_dataframe()
            for colname in list(feature_input):
                df[colname] = np.asscalar(feature_input[colname])       
                
            demfeatureData = xr.open_dataset(os.path.join(root, 'input', ROI+'_dem_features.nc'))
            demfeature_input = demfeatureData.sel(**SSA).to_dataframe()
            for colname in list(demfeature_input):
                df[colname] = np.asscalar(demfeature_input[colname])        
          
            
            df = df.reset_index(['lat','lon'])
            df.to_pickle(os.path.join(root, 'data','pickles', ROI, picklename))          
        except:
            print(str(metaAIRBASE.where(stnname,drop=True)['AirQualityStationEoICode'][0].values))
            print('file not saved')
    else:
        print(filename)
        
             
# %% load data files of meteological data, no2 sentinal observation, other time invariant 


global metaAIRBASE,lon,lat,regionrange, ROI,root

root = '.'
ROI = 'ROI1'
# loading time invariant data and airbase station information
tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
metaBase = pd.read_csv(os.path.join(root, 'input','AIRBASE', 'metadata_AIRBASE.csv'))
metaAIRBASE = xr.Dataset.from_dataframe(metaBase)
AIRBASEfilenames = glob.glob(os.path.join(root, 'data','AIRBASE','no2_'+ROI,'*.csv'))
lat = tinvData.lat
lon = tinvData.lon
regionrange = 0.001

#AIRBASEfilenames = AIRBASEfilenames[:24]
from joblib import Parallel, delayed
Parallel(n_jobs=-1)(delayed(save_data_from_near_stn)(fn) for fn in AIRBASEfilenames)


