# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:10:54 2019
Download sentinel 5p NO2 vertical column density and process 
after the process, delete files 
@author: kim
"""
from datetime import datetime, timedelta
import argparse
import os
import textwrap
import glob
import sys

import matplotlib
matplotlib.use('agg')
import numpy as np
import xarray

from amrs import models
import omi
import netCDF4
import pandas as pd

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
        
def read_geolocation(filename):

    data = xarray.Dataset()

    group = 'PRODUCT/SUPPORT_DATA/GEOLOCATIONS'
    with xarray.open_dataset(filename, group=group) as nc_file:
        data['lonc'] = nc_file['longitude_bounds'][0].copy()
        data['latc'] = nc_file['latitude_bounds'][0].copy()

    # transpose corners
    data['lonc'] = data['lonc'].transpose('corner', 'scanline', 'ground_pixel')
    data['latc'] = data['latc'].transpose('corner', 'scanline', 'ground_pixel')

    return data

def read_data(filename):
    """
    Read Tropomi NO2 fields
    """
    data = xarray.Dataset()

    with xarray.open_dataset(filename) as nc_file:
        data.attrs['time_reference'] = nc_file.time_reference
        data.attrs['orbit'] = nc_file.orbit

    with xarray.open_dataset(filename, group='PRODUCT') as nc_file:
        data['time_utc'] = nc_file['time_utc'][0].copy().astype('datetime64[ns]')
        data['vcd'] = nc_file['nitrogendioxide_tropospheric_column'][0].copy()
        data['vcd_std'] = nc_file['nitrogendioxide_tropospheric_column_precision'][0].copy()
        data['qa_value'] = nc_file['qa_value'][0].copy()

    return data

def process_data(filename, domain, qa_thr):   

 # read Level-2 data and create Level-3 grid
    grid = omi.create_grid(domain.startlon, domain.startlat, domain.stoplon,domain.stoplat, domain.dlon, domain.dlat)
    geo_data = read_geolocation(filename)

    # remove data
    mask = omi.mask_grid_domain(grid, geo_data.lonc, geo_data.latc)

    if not np.any(mask):
        return geo_data, grid
    
    # read observations
    data = read_data(filename)
    data.update(geo_data)

    # clip orbit
    data = omi.clip_orbit(grid, data=data, domain=mask)
   
    # Prepare values, errors and weigths
    rho = np.array(data['vcd'])
    errors = np.array(data['vcd_std'])
    weights = np.ones_like(rho)
    missing_values = np.array(data['qa_value'] < qa_thr)

    # grid data
    grid = omi.cvm_grid(grid, np.array(data['lonc']), np.array(data['latc']),
                        rho, errors, weights, missing_values)

    grid['time'] = data['time_utc'].astype('int64').mean().astype('datetime64[ns]').values

    
    return data, grid

         
def download_process(start, stop, domain, latt, lont, qa_thr):
    
    filename = 'tropomi_no2_%Y%m%d.tar'
    root = '.'

    for date in iter_dates(start, stop):
    
        link = '/'.join([
            'http://www.temis.nl/airpollution/no2col/data/tropomi',
            date.strftime('%Y'), date.strftime('%m'), date.strftime(filename)
        ])
    
        target = os.path.join(root, 'data', 'Sentinelno2', date.strftime(filename))
    
        print(date.strftime('%Y-%m-%d'))
    
        # download
        command = ' '.join([
            'wget', '-q', '-r', '-l1', '-nd', '-nc', '--no-parent',
            '--directory-prefix=%s' % os.path.dirname(target), link
        ])
        os.system(command)
 
        # untar
        os.system('tar -xf %s -C %s' % (target, os.path.dirname(target)))
    
        # remove tar file
        os.remove(target)
        
        PATH = "./data/Sentinelno2/S5P_OFFL*"
        filenames = glob.glob(PATH)
        
        no2 = []
        for fn in filenames:
            data, grid = process_data(fn, domain, qa_thr)       
            if not grid['values'].sum() == 0:
                no2.append(grid)
            os.remove(fn)

        no2 = xarray.concat(no2, dim='time')
        t_avg = no2['time'].astype('int64').mean().astype('datetime64[ns]').values
        no2 = no2.mean(dim='time',skipna=True)
        no2 = no2.reindex(lat=latt,lon=lont,method='nearest')        
        nc_filename = r'./data/Sentinelno2/2018/S5P_NO2_' + pd.to_datetime(t_avg).strftime('%Y-%m-%d') + '.nc'
        no2.to_netcdf(nc_filename,'w')

        
# %% main function
def main():
        
    description = textwrap.dedent("""\
    
        Python script for downloading Sentinel 5p data (Level 2),
        from 'http://www.temis.nl/airpollution/'
        
        This script is modified from a script, domino.py by Gerrit Kuhlmann
        (https://gitlab.empa.ch/abt503/users/kug/satdownload/blob/master/domino.py)
        
        by Minsu Kim (minsu.kim@empa.ch)
    """)
        
    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('starttime', type=str, help='start date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('stoptime', type=str, help='stop date (YYYY-mm-dd or YYYY-jjj)')   
    parser.add_argument('--prefix', default='.', type=str, help='root folder')
    
    args = parser.parse_args()
    
    start = parse_date(args.starttime)
    stop = parse_date(args.stoptime)
    
    DOMAIN_ALPINE = models.cosmo.Domain('Alpine', 6, 42, 12, 48,
                                         ie=4686, je=4686, pollon=180.0, pollat=90.0)
 
    root = '.'
    tinvData = xarray.open_dataset(os.path.join(root, 'input', 'ROI1_v2.nc'))
    latt = tinvData.lat
    lont = tinvData.lon
    
    qa_thr = 0.5
    
    download_process(start, stop, DOMAIN_ALPINE, latt, lont, qa_thr)

if __name__ == '__main__':
    main()

