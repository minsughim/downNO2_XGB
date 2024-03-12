# -*- coding: utf-8 -*-
"""
This script generates a netCDF of features realted to Digital Elevation Model (dem).
The features are the reconstructed dem with wavelet (here, Daubechies wavelet 2, db2, has been used as a means of multiscale analysis) that are high-pass filtered
(Ideally, fractional spline wavelet tansform can be used (assuming that geomorphology is self-affine. However, the default pywavelet does not include this wavelet, thus use db2. To do this, a customised wavelet is necessary.)
We use the levels of 10, 8, 6, and 4 for the features in this script, with the feature name of dem_hpf_1, 2, 3, and 4, respectively.
@author: kim (minsu.kim@empa.ch)
"""
# %% import packages

import xarray as xr
import os
import numpy as np 
import pywt
from dask.diagnostics import ProgressBar    
    
# %% functions

def get_new_index_for_wt(ROI):
    if ROI == 'ROI1':
        latt = np.linspace(41.95,48.05,5000)
        lont = np.linspace(5.95,12.05,5000)        
    else:
        latt = np.linspace(47.95,54.05,5000)    
        lont = np.linspace(1.95,8.05,5000)                    
    return latt, lont

def get_target_index(ROI):
    tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
    lon = tinvData.lon
    lat = tinvData.lat   
    return lat, lon

# %%
root = '/scratch/snx3000/minsukim/'
ROIlist = ['ROI1', 'ROI2']
for ROI in ROIlist: 
    
    demROI = xr.open_dataset(os.path.join(root, 'input', 'eu_dem_'+ROI+'.nc'))
    dem = demROI.Band1
    latt, lont = get_new_index_for_wt(ROI)
    # reindes the original digital elevation model as even number (nececsary for wavelet transform and reconstruction): about 0.0012 degree (100m scale)
    dem = dem.reindex(lat=latt,lon=lont,method='nearest')
    dem.chunk()
    resultswt = []

    for level in range(10):
        coeffs = pywt.wavedec2(dem, wavelet='db2', level=level)
        coeffs[0]= np.zeros_like(coeffs[0])
        filtered = pywt.waverec2(coeffs,'db2') 
        resultswt.append(filtered)

    ds = xr.Dataset({'dem_hpf_1':(['lat','lon'],resultswt[9]), # about 100km scale
                     'dem_hpf_2':(['lat','lon'],resultswt[7]), # about 25km scale
                     'dem_hpf_3':(['lat','lon'],resultswt[5]), # about 6km scale 
                     'dem_hpf_4':(['lat','lon'],resultswt[3])},coords={'lon':lont, 'lat':latt})
   
    chunks = {'lon':256, 'lat':256}
    lat, lon = get_target_index(ROI)
    ds = ds.reindex(lat=lat,lon=lon,method='nearest')
    ds = ds.chunk(chunks=chunks) 

    with ProgressBar():
        ds.to_netcdf(os.path.join(root, 'input',ROI+r'_dem_features.nc'))    
 