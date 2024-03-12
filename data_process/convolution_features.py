# -*- coding: utf-8 -*-
"""
This script generates colution features of emission sources (here, tno emission inventory, traffic volume, road length density, and population size)
Generated features are diffusion related, as it uses Gaussin convolution at different sigmas (arrival time of the diffusables at the given rate)
Two features, max_arrival_dist is the sigma value that was necessary to acheive maximum of concentraion
max_arrival_values is the maximum values that can occur at the position 

Method: Using gaussian convolutions of distribution of diffusables, spatial factors are included:  
(1) the sigma value (or time to arrival) that gives the local maxima of each location
(2) the value of convoluted density of population at the maximum 

All newly calculated features are combined in a new dataset, ROI_featues.nc

@author: kim (minsu.kim@empa.ch)
"""
import xarray as xr
import os
import numpy as np 
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy
from dask.diagnostics import ProgressBar
from distributed import Client
#client = Client()
client = Client(scheduler_file=os.environ['DASK_SCHED_FILE'])

# %%functions

def def_bbox(ROI):  
    if ROI == 'ROI1':    
        bbox = [6,12,42,48] # Alpine (around Switzerland, northen Italy, alpine area)
    else:
        bbox = [2,8,48,54]   # Benelux (northern europe)
    return bbox 

def extract_spatial_feature(inputM, max_sigma):

    inputM_g = inputM 
    inputM = client.scatter(inputM, broadcast=True)
    futures =[client.submit(gaussian_filter,inputM, sigma) for sigma in range(max_sigma)]     
    for future in futures:
        inputM_g = np.dstack((inputM_g,future.result()))   

    temp = np.argmax(inputM_g, axis=2)
    temp2 = np.amax(inputM_g, axis=2) 
    temp[temp2==0]= max_sigma 

    return temp, temp2 


def plot_distribution_of_NOX_stns(dsvar):   

    plt.figure(figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'))
    ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))#, alpha= 0.5)
    ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'))
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), linestyle=':')
    ax.add_feature(cartopy.feature.LAKES.with_scale('50m'))
    ax.add_feature(cartopy.feature.RIVERS.with_scale('50m'))
    ax.set_extent(def_bbox(ROI))    
    if dsvar.dtype == 'int64':
        plt.pcolormesh(dsvar.lon[::],dsvar.lat[::],dsvar[::,::],cmap='jet')
    else:
        plt.pcolormesh(dsvar.lon[::],dsvar.lat[::],dsvar[::,::],cmap='Spectral_r',norm=mpl.colors.LogNorm(vmin=0.0000001))
    plt.colorbar()
    plt.title(dsvar.name)
    plt.savefig(os.path.join(root, 'input','features', ROI+r'_' +dsvar.name+'.png'), dpi=300)
    

# %%
root = '/scratch/snx3000/minsukim/'

ROIlist = ['ROI1', 'ROI2']
max_sigma = 100 

for ROI in ROIlist:    
    
        # tno point sources
    if not os.path.isfile(os.path.join(root, 'input',ROI+r'_tno_feature.nc')):
        tnosourcedist = xr.open_dataset(os.path.join(root, 'input',ROI+r'_TNOsource_info.nc'))
        lon = tnosourcedist.lon
        lat = tnosourcedist.lat
        tnoemission = tnosourcedist['NOX_emission']
        tnoemission_g = tnoemission
        inputM = tnoemission
        inputM = np.nan_to_num(inputM)
        temp, temp2 = extract_spatial_feature(inputM, max_sigma)
        ds = xr.Dataset({'max_arrival_dist':(['lat','lon'],temp),'max_arrival_values':(['lat','lon'],temp2)},coords={'lon':lon, 'lat':lat})
        ds.to_netcdf(os.path.join(root, 'input',ROI+r'_tno_feature.nc'))  
        print(ROI+'_tno file created')
    else:
        print(ROI+'_tno file exists')
        
        # traffic volume
    if not os.path.isfile(os.path.join(root, 'input',ROI+r'_tfv_feature.nc')):
        tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
        lon = tinvData.lon
        lat = tinvData.lat
        inputM = tinvData.tfv
        inputM = np.nan_to_num(inputM)
        temp, temp2 = extract_spatial_feature(inputM, max_sigma)
        ds = xr.Dataset({'max_arrival_dist':(['lat','lon'],temp),'max_arrival_values':(['lat','lon'],temp2)},coords={'lon':lon, 'lat':lat})
        ds.to_netcdf(os.path.join(root, 'input',ROI+r'_tfv_feature.nc'))    
        print(ROI+'_tfv file created')
    else:
        print(ROI+'_tfv file exists')
        
        # road length density
    if not os.path.isfile(os.path.join(root, 'input',ROI+r'_rld_feature.nc')):
        tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
        lon = tinvData.lon
        lat = tinvData.lat
        inputM =  tinvData.rld
        inputM = np.nan_to_num(inputM)
        temp, temp2 = extract_spatial_feature(inputM, max_sigma)
        ds = xr.Dataset({'max_arrival_dist':(['lat','lon'],temp),'max_arrival_values':(['lat','lon'],temp2)},coords={'lon':lon, 'lat':lat})
        ds.to_netcdf(os.path.join(root, 'input',ROI+r'_rld_feature.nc'))    
        print(ROI+'_rld file created')
    else:
        print(ROI+'_rld file exists')
        
        # population density
    if not os.path.isfile(os.path.join(root, 'input',ROI+r'_pop_feature.nc')):
        tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
        lon = tinvData.lon
        lat = tinvData.lat
        inputM =  tinvData.pop
        inputM = np.nan_to_num(inputM)
        temp, temp2 = extract_spatial_feature(inputM, max_sigma)
        ds = xr.Dataset({'max_arrival_dist':(['lat','lon'],temp),'max_arrival_values':(['lat','lon'],temp2)},coords={'lon':lon, 'lat':lat})
        ds.to_netcdf(os.path.join(root, 'input',ROI+r'_pop_feature.nc'))    
        print(ROI+'_pop file created')
    else:
        print(ROI+'_pop file exists')
        
    # combine all the features to one         
    chunks = {'lon':256, 'lat':256}
    tnof = xr.open_dataset(os.path.join(root, 'input',ROI+r'_tno_feature.nc'),chunks=chunks)    
    tfvf = xr.open_dataset(os.path.join(root, 'input',ROI+r'_tfv_feature.nc'),chunks=chunks)    
    rldf = xr.open_dataset(os.path.join(root, 'input',ROI+r'_rld_feature.nc'),chunks=chunks)    
    popf = xr.open_dataset(os.path.join(root, 'input',ROI+r'_pop_feature.nc'),chunks=chunks)    
    
    ds = xr.Dataset({
            'tnof_dist':tnof.max_arrival_dist,
            'tnof_max':tnof.max_arrival_values,
            'tfvf_dist':tfvf.max_arrival_dist,
            'tfvf_max':tfvf.max_arrival_values, 
            'rldf_dist':rldf.max_arrival_dist,
            'rldf_max':rldf.max_arrival_values, 
            'popf_dist':popf.max_arrival_dist,
            'popf_max':popf.max_arrival_values         
            })
    
    urban_info = xr.open_dataset(os.path.join(root, 'input',ROI+r'_urban_info.nc'),chunks=chunks)
    ds = ds.merge(urban_info)
    
    with ProgressBar():
        ds.to_netcdf(os.path.join(root, 'input',ROI+r'_features.nc'))
        
# %% save figures of features for two ROIs
        
ROIlist = ['ROI1', 'ROI2']

for ROI in ROIlist:         
     ds = xr.open_dataset(os.path.join(root, 'input',ROI+r'_features.nc'))
     featureList = list(ds)
     for featurename in featureList:
         plot_distribution_of_NOX_stns(ds[featurename])