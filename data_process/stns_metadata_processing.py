# -*- coding: utf-8 -*-
"""
This script combines nox airbase station metadata and nabel station data and harmonisise/process for station classification.
The output file is given in csv form including meta information of stations 
(station name, type, zone, lon, lat, altitude, tinvdata, distance to urban, weighed emission sources contribution etc.)
@author: kim (minsu.kim@empa.ch)
"""
# general
import xarray as xr
import pandas as pd
import os
import numpy as np
# plotting related
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

# %% functions

def get_NOx_stations_list(df):
    
    listno2 = df.groupby('AirPollutant').get_group('NO2').AirQualityStationEoICode.unique()
    listno = df.groupby('AirPollutant').get_group('NO').AirQualityStationEoICode.unique()
    listnox = df.groupby('AirPollutant').get_group('NOX as NO2').AirQualityStationEoICode.unique()
    alllist = set().union(listno2, listno, listnox)

    shortdf = pd.DataFrame()

    for stnname in list(alllist):
        shortdf = shortdf.append(df.loc[df['AirQualityStationEoICode'] == stnname].iloc[0])        
    return pd.DataFrame(shortdf)

def def_bbox(ROI):  
    if ROI == 'ROI1':    
        bbox = [6,12,42,48] # Alpine (around Switzerland, northen Italy, alpine area)
    else:
        bbox = [2,8,48,54]   # Benelux (northern europe)
    return bbox 

def mask_ROI(df,ROI):    
    bbox = def_bbox(ROI)
    bROI = (df.Longitude>bbox[0])&(df.Longitude<bbox[1])&(df.Latitude>bbox[2])&(df.Latitude<bbox[3])    
    return df[bROI]

def read_airbase_stn_metadata(root):
    file = os.path.join(root, 'input','AIRBASE','metadata_AIRBASE.csv')
    metaBase = pd.read_csv(file)
    dflist1 = pd.DataFrame(metaBase)
    shortdf = get_NOx_stations_list(dflist1)
    airbasestns = pd.DataFrame(data={'AirQualityStationNatCode':shortdf['AirQualityStationNatCode'],
                                     'AirQualityStationEoICode':shortdf['AirQualityStationEoICode'],
                                     'AirQualityStationArea':shortdf['AirQualityStationArea'],
                                     'AirQualityStationType':shortdf['AirQualityStationType'],
                                     'Longitude':shortdf['Longitude'],
                                     'Latitude':shortdf['Latitude'],
                                     'Altitude':shortdf['Altitude']})
    return airbasestns.reset_index(drop=True)

def read_nabel_stn_metadata(root):
    dflist = pd.read_csv(os.path.join(root, 'input','NABEL','stationsliste.csv'))
    nabelstns = pd.DataFrame(data={'AirQualityStationNatCode':dflist['IDB-code'],
                                   'AirQualityStationEoICode':dflist['EoI-code'],
                                   'AirQualityStationArea':dflist['Type of zone'],
                                   'AirQualityStationType':dflist['Type of station'],
                                   'Longitude':dflist['m_lon'],
                                   'Latitude':dflist['m_lat'],
                                   'Altitude':dflist['m_altitude']})
    temp = list(set(nabelstns.AirQualityStationEoICode.unique()).intersection(airbasestns.AirQualityStationEoICode.unique()))

    for stnname in temp:
        tempr = nabelstns[nabelstns.AirQualityStationEoICode==stnname]
        nabelstns = nabelstns.drop(tempr.index, axis=0)  

    for i in np.arange(0,len(nabelstns),1):
        if pd.isnull(nabelstns.AirQualityStationEoICode.iloc[i]):
            nabelstns.AirQualityStationEoICode.iloc[i] = 'CH'+ nabelstns.AirQualityStationNatCode.iloc[i]
    return nabelstns.reset_index(drop=True)


def plot_distribution_of_stns(stndata,ROI):   
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND.with_scale('50m'))
    ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'))
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), linestyle=':')
    ax.add_feature(cartopy.feature.LAKES.with_scale('50m'))
    ax.add_feature(cartopy.feature.RIVERS.with_scale('50m'))
    ax.set_extent(def_bbox(ROI))    
    plt.plot(stndata.Longitude, stndata.Latitude,'mo', markeredgecolor='k', markersize=4)
    plt.legend()
    return fig

# %% read metadata of stations and combine to one
root = '/scratch/snx3000/minsukim/'
airbasestns = read_airbase_stn_metadata(root)
nabelstns = read_nabel_stn_metadata(root)
df = airbasestns.append(nabelstns)
df = df.reset_index(drop=True)

# %% Main:: adding information of time invariance data for all stations

ROIList = ['ROI1', 'ROI2']

for ROI in ROIList:
    tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))
    featureData = xr.open_dataset(os.path.join(root, 'input', ROI+'_features.nc'))
    lat = tinvData.lat
    lon = tinvData.lon
    
    stndata = pd.DataFrame()
    stns_ROI = mask_ROI(df,ROI)
    
    for i in np.arange(0,len(stns_ROI),1):
        df1 = stns_ROI.iloc[i]
        
        time_inv_input = tinvData.sel(lat=df1.Latitude, lon=df1.Longitude, method='nearest')
        for colname in list(time_inv_input):
            df1[colname] = np.asscalar(time_inv_input[colname])          
       
        feature_input = featureData.sel(lat=df1.Latitude, lon=df1.Longitude, method='nearest')
        for colname in list(feature_input):
            df1[colname] = np.asscalar(feature_input[colname])     
                                
        stndata = stndata.append(df1)

    stndata = stndata.reset_index(drop=True)    
    stndata.to_csv(os.path.join(root,'input',ROI +'_stn_metadata.csv'))
    plot_distribution_of_stns(stndata,ROI)
