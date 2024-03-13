# -*- coding: utf-8 -*-
"""
Creating spatial data with gaussian convolution with different sigmas
population density, road lenth density, and traffic volumes are convoltued for different scales, 1,2,4,8, 16 etc. (1 = 100m) 
These variables are a proxy of emission sources

Reference: 
Minsu Kim, Dominik Brunner, Gerrit Kuhlmann (2021) 
Importance of satellite observations for high-resolution mapping of near-surface NO2 by machine learning, 
Remote sensing of Environment DOI: https://doi.org/10.1016/j.rse.2021.112573

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""
import scipy.ndimage as ndi
import xarray as xr
import os
import numpy as np

# %% functions
def def_bbox(ROI):  
    if ROI == 'ROI1':    
        bbox = [6,12,42,48] # Alpine (around Switzerland, northen Italy, alpine area)
    else:
        bbox = [2,8,48,54]   # Benelux (northern europe)
    return bbox 


# %%
root = '.'
ROI = 'ROI1'
# loading time invariant data
tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
lat = tinvData.lat
lon = tinvData.lon


ds = xr.Dataset()
for i in np.arange(0,10,1):
    x = tinvData.pop.values
    sd = 2**i
    xg = ndi.gaussian_filter(x,sd)
    ds['pop_%d'% sd] = xr.DataArray(xg, coords={'lat': lat, 'lon': lon},dims=['lat', 'lon'])

    x = tinvData.rld.values
    x[np.isnan(x)]=0
    xg = ndi.gaussian_filter(x,sd)
    ds['rld_%d'% sd] = xr.DataArray(xg, coords={'lat': lat, 'lon': lon},dims=['lat', 'lon'])
    
    x = tinvData.tfv.values
    x[np.isnan(x)]=0
    xg = ndi.gaussian_filter(x,sd)
    ds['tfv_%d'% sd] = xr.DataArray(xg, coords={'lat': lat, 'lon': lon},dims=['lat', 'lon'])
        
ds.to_netcdf(os.path.join(root, 'data',ROI+r'_tinvData_gaussian.nc'))

