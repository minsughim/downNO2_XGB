# -*- coding: utf-8 -*-
"""
This script is to add Swiss BAFU NOx and NO2 data (2018 only) to the main dataset
The staion names are reassigned as "CH<IDB-code>" (e.g. CHagAAR)
After processing the data, time series of each station is pickled. 
At the end of the scrtipt, NABEL data and AIRBASE data are combined for ROI1 (Alpine region)
@author: kim (minsu.kim@empa.ch)
"""

# %% import packages
import xarray as xr
import pandas as pd
import glob
import numpy as np
import os

# %% functions 

def find_stn_location(stnname):
    temp = dfstn['IDB-code']==stnname
    stnlon = dfstn.where(temp,drop=True).m_lon
    stnlat = dfstn.where(temp,drop=True).m_lat
    stnlon = lon.sel(lon=stnlon, method='nearest') #find nearest point for the selection of spatial domiain
    stnlat = lat.sel(lat=stnlat, method='nearest')
    return [np.asscalar(stnlat) , np.asscalar(stnlon)]

def read_ERA5_time_series_data(varname, root, SSA):
	filenames = 'meteo_interp_' +varname+'_*.nc'
	all_files = sorted(glob.glob(os.path.join(root, 'data','ERA5', ROI, filenames)))
	all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
	ds_concat = xr.concat(all_dsets, dim='time')
	return ds_concat.sel(**SSA).to_dataframe()

def read_NO2_time_series_data(root, SSA):
	all_files = sorted(glob.glob(os.path.join(root, 'data','Sentinelno2',ROI,'S5P_NO2_*.nc')))
	all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
	ds_concat = xr.concat(all_dsets, dim='time')
	return ds_concat.sel(**SSA).to_dataframe()	

def save_data_from_near_stn(stnname):    
    no2Timeseries = pd.DataFrame(data={'time':nabeldf['Unnamed: 0'],'stringC':nabeldf[stnname]})
    no2Timeseries = no2Timeseries.set_index('time', drop=True)
    no2Timeseries = no2Timeseries.drop(no2Timeseries.index[:8])
    no2Timeseries['Concentration'] = no2Timeseries.stringC.str.replace(',','.').astype(float)
    del no2Timeseries['stringC'] 
    
    picklename =  'CH'+stnname+'_2018_2018'+'_features.pkl'
    root = '/scratch/snx3000/minsukim/'
    target = os.path.join(root, 'data','pickles', ROI, picklename)
    if not os.path.isfile(target): 
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
    
            met_stn_series = met_stn_series.reset_index().set_index('time')
    
            mfdsNo2 = mfdsNo2.reset_index().set_index('time')
    
            df1 = pd.merge(no2Timeseries,met_stn_series,left_index=True,right_index=True,how='inner') #(default) merge only rows existing in df (left)
            df = pd.merge(df1,mfdsNo2,left_index=True,right_index=True,how='outer') # merge all rows and fill with nan
            
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
            
            stnclassesData = xr.open_dataset(os.path.join(root, 'input', ROI+'_stn_classes.nc'))
            stnclass_input = stnclassesData.sel(**SSA).to_dataframe()
            for colname in list(stnclass_input):
                df[colname] = np.asscalar(stnclass_input[colname])     

            df.to_pickle(target)          
        except:
            print(stnname)
            print('file not saved')
    else:
        print(picklename)
        

# %% 
root = '/scratch/snx3000/minsukim/'

global lon,lat,regionrange, ROI, nabeldf, dfstn
ROI = 'ROI1'

dflist = pd.read_csv(os.path.join(root, 'input','NABEL','stationsliste.csv'))
dfstn = xr.Dataset.from_dataframe(dflist)

# loading time invariant data and airbase station information
tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
lat = tinvData.lat
lon = tinvData.lon
regionrange = 0.001

nabeldf = pd.read_excel(os.path.join(root, 'input','NABEL','NO2_2018_EMPA_Kim.xlsx'))
stnnamelists = list(nabeldf)
stnnamelists = stnnamelists[1:]

from joblib import Parallel, delayed
Parallel(n_jobs=-1)(delayed(save_data_from_near_stn)(stnname) for stnname in stnnamelists)
