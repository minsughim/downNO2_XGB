# -*- coding: utf-8 -*-
"""
Python script for reindexing Sentinel 5p data (Level 2),
The target grid is given from land use data of CORINE2018 (100m scale)
For imputed data, such as nan, qa vaules below 0.5 are assigned nan
For interpolation, 'nearest' method is used

Reference: 
Minsu Kim, Dominik Brunner, Gerrit Kuhlmann (2021) 
Importance of satellite observations for high-resolution mapping of near-surface NO2 by machine learning, 
Remote sensing of Environment DOI: https://doi.org/10.1016/j.rse.2021.112573

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""

import os
import xarray as xr
import numpy as np
import scipy.interpolate as intp
from datetime import datetime, timedelta
import argparse
import textwrap
import glob

# %% functions
def parse_date(s):
    if len(s) == 10:
        return datetime.strptime(s, '%Y-%m-%d')
    else:
        return datetime.strptime(s, '%Y-%j')

def iter_dates(start, stop):
    current = start
    while current <= stop:
        yield current
        current += timedelta(days=1)
        
def ROI_mask_for_regridding(lat,lon):
    thres = 0.05
    maskROI = [lon.min()-thres, #boundary padding for regridding, padding siye determined as thres. NOTE: this does not affect the value at the boundary
               lon.max()+thres,
               lat.min()-thres,
               lat.max()+thres]        
    return maskROI

def reindex_sentinel(start, stop, ROI, root='.'):

    # import land usage data to get the grid for calculation (100 meter scale)
    # When the regridding based on the desired resolution.
    #lat = xr.Variable('lat', np.arange(lat1,lat2,latDelta))
    #lon = xr.Variable('lon', np.arange(lon1,lon2,lonDelta))    
    # Land usage data as a basis for the resolution, reindex every other dataset based on this
    filenc = './input/LUD/g100_clc18_'+ROI+'.nc' #The input file can be downloaded from https://polybox.ethz.ch/index.php/s/H17MpAVwZqsZZVm/download
    dsROI = xr.open_dataset(filenc)
    #bring lat, lojn distribution based from land usage data
    lat = dsROI.lat
    lon = dsROI.lon
    lonm, latm = np.meshgrid(lon,lat) 
    maskROI = ROI_mask_for_regridding(lat,lon)
    datestr = '%Y%m%d' # Sentinel data: processing for a day
    
    for date in iter_dates(start, stop):
        print(date.strftime(datestr))
        no2 =[]
        pathname = './data/Sentinelno2/S5P_OFFL_L2__NO2____'+ date.strftime(datestr)+'*.nc'

        filenames = glob.glob(pathname)
        for f in filenames:
            ds = xr.open_dataset(f, group='PRODUCT').rename({'longitude':'lon','latitude':'lat'})
            bbROI = (ds.lon>maskROI[0])&(ds.lon<maskROI[1])&(ds.lat>maskROI[2])&(ds.lat<maskROI[3])&(ds.qa_value>0.5)&(ds.nitrogendioxide_tropospheric_column>0) # filter our low quality and negative values
            dsROI = ds.where(bbROI,drop=True)
            if dsROI.lon.size != 0 :
                x1 = dsROI.lon[0,:,:].values.ravel()
                y1 = dsROI.lat[0,:,:].values.ravel()
                no2_original = dsROI.nitrogendioxide_tropospheric_column[0,:,:].values.ravel()
                points1 = np.column_stack((x1,y1))
                regridded_no2 = intp.griddata(points1, no2_original, (lonm,latm), method='nearest')
                regridded_no2_1 = intp.griddata(points1, no2_original, (lonm,latm), method='linear')
                regridded_no2[np.isnan(regridded_no2_1)] = np.nan
                no2_o = xr.DataArray(regridded_no2, coords=[lon, lat], dims=['lon', 'lat'])
                no2.append(no2_o)
                
        no2 = xr.concat(no2, dim='time')
        no2 = no2.chunk(chunks={'lon':256, 'lat':256})   
        no2 = no2.mean(dim='time',skipna=True) 
        ds = no2.to_dataset(name='no2')
        newfilename = r's5p_no2_'+ROI+'_'+date.strftime(datestr)+'.nc'
        target = os.path.join(root, 'Sentinelno2', newfilename)
        ds.to_netcdf(target)

def main():
    
    root = './data/'
    
    description = textwrap.dedent("""\
    
        Python script for reindexing Sentinel 5p data (Level 2),
        The target grid is given from land use data of CORINE2018 (100m scale)
        For imputed data, such as nan, qa vaules below 0.5 are assigned nan
        For interpolation, 'nearest' method is used
        
        For regridding land usage dataset gridding is used as the target grid 
        the example file for this script can be downloaded from
        https://polybox.ethz.ch/index.php/s/H17MpAVwZqsZZVm/download
        
        by Minsu Kim (minsu.kim@empa.ch)
    """)
        
    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('starttime', type=str, help='start date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('stoptime', type=str, help='stop date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('ROI', type=str, help='region of interest (type ROI1 or ROI2 -- ROI1: Alpine or ROI2:Benelux)')
    
    # If the land usage data (100m resolution) is the target resoltuion, land usage data for the selected region should be rewrapped.

    parser.add_argument('--prefix', default='.', type=str, help='root folder')
    
    args = parser.parse_args()
    
    start = parse_date(args.starttime)
    stop = parse_date(args.stoptime)
    ROI = args.ROI
    
    reindex_sentinel(start, stop, ROI=ROI, root=root)

if __name__ == '__main__':
    main()
