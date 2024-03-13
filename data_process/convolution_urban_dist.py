# -*- coding: utf-8 -*-
"""
This script is to produce urban analysis results in a netCDF file
The results include 
1. original smod (settlement model) data
2. the shortest distance to a urban boundary 
3. the shortest distance to a city boundary 
@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""
import xarray as xr
from scipy import ndimage
import os

# %% functions 

def load_urban_info(ROI):
    
    if not os.path.isfile(os.path.join(root, 'input',ROI+r'_urban_info.nc')): 
        # loading time invariant data
        tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
        lat = tinvData.lat
        lon = tinvData.lon
        da = xr.open_dataset(os.path.join(root, 'input', 'SMOD','smod_'+ROI+'.nc'))
        da = da.reindex(lat=lat,lon=lon,method='nearest')
        # here 1: rural, 2: urban clusters (low density) 3: urban centers (high density)
        urbanmask = (da.Band1>1)
        citymask = (da.Band1==3)
    
        dist_to_urban = ndimage.distance_transform_edt(~urbanmask)
        dist_to_city = ndimage.distance_transform_edt(~citymask)
    
        # reindex urban indices
        ds = xr.open_dataset(os.path.join(root, 'input', 'SMOD','urban_'+ROI+'.nc'))
        ds = ds.reindex(lat=lat,lon=lon,method='nearest')
    
        # save all information in netCDF
        urban_info = xr.Dataset({'urban_reindexed':(['lat','lon'],ds.Band1),'smod':(['lat','lon'],da.Band1)},coords={'lon':lon, 'lat':lat})  
        urban_info['dist_urban'] = xr.DataArray(dist_to_urban, coords={'lat': lat, 'lon': lon},dims=['lat', 'lon'])
        urban_info['dist_city'] = xr.DataArray(dist_to_city, coords={'lat': lat, 'lon': lon},dims=['lat', 'lon'])
    
        chunks = {'lon':256, 'lat':256}
        urban_info = urban_info.chunk(chunks=chunks)
        urban_info.to_netcdf(os.path.join(root, 'input',ROI+r'_urban_info.nc'))           
    else:
        # load processed arrays        
        urban_info = xr.open_dataset(os.path.join(root, 'input',ROI+r'_urban_info.nc'))
    return urban_info

# %% creating settlement mode information and diffusable components analysis of population density ==== making a one netcdf with five variables. 
root = '.'

ROIList = ['ROI1', 'ROI2']
for ROI in ROIList:
    urban_info = load_urban_info(ROI)