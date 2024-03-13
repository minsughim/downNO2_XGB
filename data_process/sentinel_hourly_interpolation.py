# -*- coding: utf-8 -*-
"""
This script is to interpolate daily satelliate observation (TROPOMI, Sentinel 5p) to hourly (temporal linear interpolation)
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
import os

# %% functions

def read_NO2_time_series_data(root, ROI):
	all_files = glob.glob(os.path.join(root, 'data','Sentinelno2',ROI,'S5P_NO2_*.nc'))
	all_dsets = [xr.open_dataset(fname).chunk() for fname in all_files]
	return xr.concat(all_dsets, dim='time').sortby('time')

def temporal_interpolation_satellite_data(ds_part):
    if not os.path.isfile(os.path.join(root, 'data','Sentinelno2',ROI,'S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] +'.nc' )): 
        temptt = pd.date_range(ds_part.time[0].values, ds_part.time[-1].values, freq='H')
        temptt = xr.DataArray(temptt,dims='time')
        ds_part = ds_part.reindex_like(temptt).chunk(chunks={'time':-1,'lon':256,'lat':256})
        ds_part = ds_part.interpolate_na(dim='time', limit =24)
        ds_part[0:-1,:,:].to_netcdf(os.path.join(root, 'data','Sentinelno2',ROI,'S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] +'.nc' ))
    else:
        print('file exists : S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] +'.nc')

def temporal_interpolation_satellite_data_end(ds_part): # at the end of the time series, include the last observation to save
    if not os.path.isfile(os.path.join(root, 'data','Sentinelno2',ROI,'S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] +'.nc' )): 
        temptt = pd.date_range(ds_part.time[0].values, ds_part.time[-1].values, freq='H')
        temptt = xr.DataArray(temptt,dims='time')
        ds_part = ds_part.reindex_like(temptt).chunk(chunks={'time':-1,'lon':256,'lat':256})
        ds_part = ds_part.interpolate_na(dim='time', limit =24)
        ds_part.to_netcdf(os.path.join(root, 'data','Sentinelno2',ROI,'S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] +'.nc' ))
    else:
        print('file exists : S5P_hourly_NO2_' + str(ds_part.time[0].values)[0:10] +'.nc')

# %%
    
root = '.'
ROI = 'ROI1'
mfds = read_NO2_time_series_data(root, ROI)

from joblib import Parallel, delayed
Parallel(n_jobs=-1)(delayed(temporal_interpolation_satellite_data)(mfds.no2[t:(t+2),:,:]) for t in range(1,(len(mfds.time)-2),1))

temporal_interpolation_satellite_data_end(mfds.no2[-2:,:,:])
