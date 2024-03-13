# -*- coding: utf-8 -*-
"""
This script is to produce urban analysis results in a netCDF file
The results include 
1. original smod (settlement model) data
2. the shortest distance to a urban boundary 
3. the shortest distance to a city boundary 
4. weighed distance with the population size of each urban area (exponental decaying functions are assumed with decaying length of 100m, 1km, 5km)

NOTE: This script is computationally very expensive and (in piz daint, workers were killed due to unknown reasons), thus this script is discarded for the model
:as an alternative feature, gaussian colvolution of population distribtuion is chosen. Details of the convolution method, see convolution_urban_dist.py 

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""
import xarray as xr
from scipy import ndimage
import os
import numpy as np 
from numba import njit
import time 

from dask.distributed import as_completed
from dask.distributed import Client 
client = Client(scheduler_file=os.environ['DASK_SCHED_FILE'])

# %% functions 

@njit
def weighed_distance_map(dist_to_temp, popsize):
    return popsize*np.exp(-dist_to_temp/distance0)

@njit
def update_location(position):
    temp = np.zeros_like(dempop)
    temp[position[0],position[1]] += 1
    return temp

def numba_weighed_sum(positionlist):
    urban_weight = np.zeros_like(dempop)
    for position in positionlist:
        urban_weight += weighed_distance_map(ndimage.distance_transform_edt(update_location(position)==0), dempop[position[0],position[1]])  
    return urban_weight

def numba_weighed_sum_write(positionlist):
    urban_weight = np.zeros_like(dempop)
    for position in positionlist:
        urban_weight += weighed_distance_map(ndimage.distance_transform_edt(update_location(position)==0), dempop[position[0],position[1]])  
           
    filename = r'urban_weightdist_%d_%d_%d.npy' % (distance0, position[0],position[1])
    np.save(os.path.join(root, 'data','urban_weight',ROI, filename),urban_weight) # from june to December (sencond half of 2018 for ROI1)
    return urban_weight


def load_urbaninds_dempop(ROI):

    tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
    lat = tinvData.lat
    lon = tinvData.lon
    
    if not os.path.isfile(os.path.join(root, 'input',ROI+r'_urban_info.nc')): 
        # loading time invariant data

        da = xr.open_dataset(os.path.join(root, 'input', 'SMOD','smod_'+ROI+'.nc'))
        da = da.reindex(lat=lat,lon=lon,method='nearest')
        # here 1: rural, 2: urban clusters (low density) 3: urban centers (high density)
        ruralmask = (da.Band1==1)
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
    
        # save nonzero urbaninds positions in binary  
        urbaninds = np.nonzero(ds.Band1.data)
        np.save(os.path.join(root, 'input','urbaninds_'+ROI+'.npy'), urbaninds)
    
    else:
        # load processed arrays
        
        urban_info = xr.open_dataset(os.path.join(root, 'input',ROI+r'_urban_info.nc'))
        urbaninds = np.load(os.path.join(root, 'input','urbaninds_'+ROI+'.npy'))
    return urbaninds, tinvData['pop'].data, urban_info
        

# %% open settlement model (smod) and query for distance to urban and reindex the basic information including distance to urban and city save the info in netcdf

root = '.'
global dempop, distance0, ROI

ROIList = ['ROI1', 'ROI2']
ROI = ROIList[0]
distance0 = 10
urbaninds, dempop, urban_info = load_urbaninds_dempop(ROI)

# %% dask + numba

start = time.time()
slicesize = 100
c_lenth = 2*slicesize*20
urban_weight = np.zeros_like(dempop)
params = client.scatter({'dempop':dempop, 'distance0':distance0}, broadcast=True)
futures = [client.submit(numba_weighed_sum, zip(urbaninds[0][i:i+slicesize],urbaninds[1][i:i+slicesize])) for i in np.arange(0,c_lenth,slicesize)]
for future, results in as_completed(futures, with_results=True):
    urban_weight += results[0]

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
