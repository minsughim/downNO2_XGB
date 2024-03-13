# -*- coding: utf-8 -*-
"""
This script loads TNO point source of NOX and calcualtes distance to the closest sources and weighted sum of distances with emitted amount

Reference: 
Minsu Kim, Dominik Brunner, Gerrit Kuhlmann (2021) 
Importance of satellite observations for high-resolution mapping of near-surface NO2 by machine learning, 
Remote sensing of Environment DOI: https://doi.org/10.1016/j.rse.2021.112573

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""

import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import os

# %% functions

def def_bbox(ROI):  
    if ROI == 'ROI1':    
        bbox = [6,12,42,48] # Alpine (around Switzerland, northen Italy, alpine area)
    else:
        bbox = [2,8,48,54]   # Benelux (northern europe)
    return bbox 

def mask_ROI(df,ROI):   
    bbox = def_bbox(ROI)
    bROI = (df.lon>bbox[0])&(df.lon<bbox[1])&(df.lat>bbox[2])&(df.lat<bbox[3])
    stns_ROI = df[bROI]
    return stns_ROI[stns_ROI.NOX>0]

def read_tno_point_sources(ROI):
    df = pd.read_csv(os.path.join(root, 'TNO_6x6_GHGco_v1_1/TNO_GHGco_v1_1_year2015.csv'), sep=';',usecols=['Lon','Lat','NOX','GNFR_Sector','SourceType'],skipinitialspace=True)
    df['lon'] = df.Lon
    df['lat'] = df.Lat
    df = df.drop(columns=['Lon', 'Lat'])
    dfROI = mask_ROI(df,ROI) 
    ListPointSources = dfROI.groupby('SourceType').get_group('P')
    ListPointSources = ListPointSources.reset_index(drop=True)
    return ListPointSources
    
def make_list_of_TNO_nox_sources(ROI):
    ListPointSources = read_tno_point_sources(ROI)    
    Listsources = ListPointSources.drop(ListPointSources.groupby('GNFR_Sector').get_group('J').index) #discard Waste sources due to the lack of reports from some countries
    return Listsources.reset_index(drop=True)

def make_list_of_TNO_nox_sources_typewise(ROI, typeX):
    ListPointSources = read_tno_point_sources(ROI)
    return ListPointSources.groupby('GNFR_Sector').get_group(typeX).reset_index(drop=True)

def lon_lat_index_variables(lat, lon):
    # to calculate indexes for raster array, minimum lon., lat and delta values are obtained using this fuction
    devlon = (lon -np.roll(lon,1))
    devlon[0] = devlon[1]
    devlat = (lat - np.roll(lat,1))
    devlat[0] = devlat[1]
    Dlon = devlon.mean().values
    Dlat = devlat.mean().values
    lon0 = lon[0].values
    lat0 = lat[0].values    
    return Dlon, Dlat, lon0, lat0

def rastered_dist_to_nox_source(Listsources,ROI): #this function returns rastersied point nox emission source distribution
    dataSpatial = xr.open_dataset(os.path.join(root, 'data',ROI+r'.nc'))
    lat = dataSpatial.lat   
    lon = dataSpatial.lon
    Dlon, Dlat, lon0, lat0 = lon_lat_index_variables(lat, lon)
    NOX_point_sources = xr.Dataset({'NOX_emission':(['lat','lon'],xr.zeros_like(dataSpatial.lu))}, coords={'lon':lon, 'lat':lat})
    for i in np.arange(0,len(Listsources),1):
        latI = int(round(np.asscalar((Listsources.lat[i]-lat0)/Dlat)))
        lonI = int(round(np.asscalar((Listsources.lon[i]-lon0)/Dlon)))
        NOX_point_sources.NOX_emission[latI,lonI] += Listsources.NOX[i]        
    dist_to_source = ndi.distance_transform_edt((NOX_point_sources.NOX_emission==0))
 
    return NOX_point_sources, dist_to_source

def rastered_nox_source_with_weight(Listsources,ROI):       
    dataSpatial = xr.open_dataset(os.path.join(root, 'data',ROI+r'.nc'))
    #dataSpatial = xr.open_dataset(r'/scratch/snx3000/minsukim/input/ROI1.nc')
    lat = dataSpatial.lat   
    lon = dataSpatial.lon
    Dlon, Dlat, lon0, lat0 = lon_lat_index_variables(lat, lon)

    temp = np.zeros_like(dataSpatial.lu)
    NOX_emission = np.zeros_like(dataSpatial.lu)   
    NOX_emission_weight_100m = np.zeros_like(dataSpatial.lu)   
    NOX_emission_weight_1km = np.zeros_like(dataSpatial.lu)   
    NOX_emission_weight_5km = np.zeros_like(dataSpatial.lu)   

    for i in np.arange(0,len(Listsources),1):
        latI = int(round(np.asscalar((Listsources.lat[i]-lat0)/Dlat)))
        lonI = int(round(np.asscalar((Listsources.lon[i]-lon0)/Dlon)))
        NOX_emission[latI,lonI] += Listsources.NOX[i]        
        temp[latI,lonI] = 1
        dist_temp = ndi.distance_transform_edt(temp==0)
        temp[latI,lonI] = 0 #put it back to zero for next calculation
        NOX_emission_weight_100m += Listsources.NOX[i]*np.exp(-dist_temp)
        NOX_emission_weight_1km += Listsources.NOX[i]*np.exp(-dist_temp/10)
        NOX_emission_weight_5km += Listsources.NOX[i]*np.exp(-dist_temp/50)
        
    dist_to_source = ndi.distance_transform_edt((NOX_emission==0))

    NOX_point_sources_weighted = xr.Dataset({'NOX_emission':(['lat','lon'],NOX_emission), 
                                             'dist_to_source':(['lat','lon'],dist_to_source),
                                             'NOX_emission_weight_1km':(['lat','lon'],NOX_emission_weight_1km),
                                             'NOX_emission_weight_5km':(['lat','lon'],NOX_emission_weight_5km),
                                             'NOX_emission_weight_100m':(['lat','lon'],NOX_emission_weight_100m)}, coords={'lon':lon, 'lat':lat})          
    return NOX_point_sources_weighted

def plot_tno_point_sources_by_type(ROI):   
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'))
    ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'))
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), linestyle=':')
    ax.add_feature(cartopy.feature.LAKES.with_scale('50m'))
    ax.add_feature(cartopy.feature.RIVERS.with_scale('50m'))
    ax.set_extent(def_bbox(ROI))    
    sourcelist = ['A','J','D','H','B']
    sourcetypename = ['A: Public power','J: Waste','D: Fugitives','H: Aviation','B:Industry']
    for typeX, typecat in zip(sourcelist,sourcetypename):
        soucef = make_list_of_TNO_nox_sources_typewise(ROI, typeX)
        plt.scatter(x=soucef.lon, y=soucef.lat,s=np.log10(soucef.NOX)*15, alpha= 0.5,label=typecat)
    plt.legend()
    return fig
    
def plot_tno_distribution_sources(ROI,da):   
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'))
    ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'))
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), linestyle=':')
    ax.add_feature(cartopy.feature.LAKES.with_scale('50m'))
    ax.add_feature(cartopy.feature.RIVERS.with_scale('50m'))
    ax.set_extent(def_bbox(ROI))
    plt.pcolormesh(da.lon,da.lat,da,cmap='jet',alpha=0.5)
    plt.colormap() 
    sourcelist = ['A','D','H','B']
    sourcetypename = ['A: Public power','D: Fugitives','H: Aviation','B:Industry']    
    for typeX, typecat in zip(sourcelist,sourcetypename):
        soucef = make_list_of_TNO_nox_sources_typewise(ROI, typeX)
        plt.scatter(x=soucef.lon, y=soucef.lat,s=np.log10(soucef.NOX)*15, alpha= 0.5,label=typecat)
    plt.legend()
    return fig

# %% Main 
global root 
root = '.'

ROIList = ['ROI1', 'ROI2']

for ROI in ROIList:

    Listsources = make_list_of_TNO_nox_sources(ROI) # get location list of point sources
    plot_tno_point_sources_by_type(ROI) # simple checks ith a plot for the locations of point sources
    NOX_point_sources_weighted = rastered_nox_source_with_weight(Listsources,ROI) #get rastered arrays of tno emission data of industry, power plants etc

    chunks = {'lon':256, 'lat':256}
    NOX_point_sources_weighted = NOX_point_sources_weighted.chunk(chunks=chunks)
    NOX_point_sources_weighted.to_netcdf(os.path.join(root, 'input',ROI+r'_TNOsource_info.nc')) # save results in netCDF 
    #plot_tno_distribution_sources(ROI,NOX_point_sources_weighted.NOX_emission_weight_5km)