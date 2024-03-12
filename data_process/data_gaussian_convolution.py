# -*- coding: utf-8 -*-
"""
Creating spatial data with gaussian convolution with different sigmas
population density, road lenth density, and traffic volumes are convoltued for different scales, 1,2,4,8, 16 etc. (1 = 100m) 
These variables are a proxy of emission sources
@author: kim
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
# TODO: The boundaries are treated in a model of 'reflect'. It needs an adequte mode by applying bigger domain.
xg = ndi.gaussian_filter(elevation,5)
temp = ndi.filters.laplace(xg)
plt.imshow(temp)
plt.colorbar()

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

# %% To quickly plot the output
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
import matplotlib as mpl
import cartopy
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,4))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND.with_scale('50m'))
ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'))
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), linestyle=':')
ax.add_feature(cartopy.feature.LAKES.with_scale('50m'))
ax.add_feature(cartopy.feature.RIVERS.with_scale('50m'))
ax.set_extent(def_bbox(ROI))    
plt.pcolormesh(lon,lat,temp10,cmap='Spectral_r',alpha=0.5)
plt.colorbar()