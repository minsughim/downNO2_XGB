# -*- coding: utf-8 -*-
# %% import packages

import xarray as xr
import pandas as pd
import numpy as np
import os
import argparse
import textwrap

# %%
def interpolate_meteo(ds):   
   root = '/scratch/snx3000/minsukim/'
   tinvData = xr.open_dataset(os.path.join(root, 'input', ROI+'_v2.nc'))   
   datestr = '%Y%m%d_%p'
   date = pd.to_datetime(ds.time[0].values)
   filename = r'meteo_interp_'+ds.name+ '_' + date.strftime(datestr) + '.nc'
   target = os.path.join(root, 'data','ERA5', ROI, filename)   
   if not os.path.isfile(target): 
       ds_interp = []
       for t_obs in ds.time:
           ds_t = ds.sel(time=t_obs)
           temp = ds_t.interp_like(tinvData) 
           temp = temp.expand_dims('time')
           ds_interp.append(temp) 
           
       ds_interp = xr.concat(ds_interp, dim='time')   
       ds_interp.to_netcdf(target) # from june to December (sencond half of 2018 for ROI1)
   else:
       print('file exsits:'+target)

# %% main function
def main():
    
    description = textwrap.dedent("""\
        Python script to linearly interpolate meteological data from ERA5 (30km scale) to 100m scale (for downno2 project)
        Five meteological data for 2018 are available in this case: 't2m', 'v10', 'u10', 'cdir', 'tp', 'blh'
        The data will be interpolated and saved in two files (AM, PM) per day.
        by Minsu Kim (minsu.kim@empa.ch)
    """)
            
    root = '/scratch/snx3000/minsukim/'
    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)   
    parser.add_argument('met_var', type=str, help='meteological data: choose one of t2m, v10, u10, cdir, tp, blh')
    parser.add_argument('ROI', type=str, help='Region of interest: ROI1 or ROI2')

    args = parser.parse_args()

    global ROI 
    
    ROI = args.ROI
    mfdsMet = xr.open_dataset(os.path.join(root, 'data','ERA5', 'met_' + ROI + '.nc'))


    from joblib import Parallel, delayed
    Parallel(n_jobs=18)(delayed(interpolate_meteo)(mfdsMet[args.met_var].isel(time=slice(i,i+12))) for i in np.arange(0,len(mfdsMet.time),12))

if __name__ == '__main__':
    main()    