# -*- coding: utf-8 -*-
"""
This python script is to merge predicted maps for each hour and plot a sample figure 
While merging, the script also gathers missing parts of the results and save

Reference: 
Minsu Kim, Dominik Brunner, Gerrit Kuhlmann (2021) 
Importance of satellite observations for high-resolution mapping of near-surface NO2 by machine learning, 
Remote sensing of Environment DOI: https://doi.org/10.1016/j.rse.2021.112573

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743
"""
# %% import packages


import os
import xarray as xr
import glob
import numpy as np
import argparse
import textwrap
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
from matplotlib.colors import LightSource
from datetime import datetime, timedelta

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

def plot_save_fig(ds,fname):     

    print(fname)
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.add_feature(cartopy.feature.LAND.with_scale('50m'), color ='slategrey')
    ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'),alpha= 0.4, edgecolor='k', zorder=3)
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), alpha= 0.4, edgecolor='k',zorder=3)
    #ax.add_feature(cartopy.feature.LAKES.with_scale('50m'), alpha= 0.3)
    #ax.add_feature(cartopy.feature.RIVERS.with_scale('50m'), alpha= 0.3)
    no2 = ax.imshow(ds.no2[0,:,:], cmap=newcmp, aspect=1, vmin=0, vmax=60, extent=[6,12,42,48],zorder =1, origin='lower')
    #no2= plt.pcolormesh(ds.lon,ds.lat,ds.no2[0,:,:], cmap=newcmp, vmin = 0, vmax=60, alpha = 0.8)
    ax.imshow(rgb,  aspect=1, extent=[6,12,42,48], cmap='gray',alpha = 0.1, zorder =2,origin='lower',interpolation='None')
    ax.set_xticks(np.arange(6,13,1))
    ax.set_yticks(np.arange(42,49,1))
    ax.set_xlim(6,12)
    ax.set_ylim(44,48)
    
    cbar = fig.colorbar(no2, ax=ax, extend='max',shrink=0.75,pad=0.03,label='Near-surface NO$_2$ concentration [µg.m$^{-3}$]')
    ax.set_title(str(ds.time[0].values).replace('T', ' ')[:19], fontsize=14)
    ax.set_xlabel('Longitude (°E)',fontsize=12)
    ax.set_ylabel('Latitude (°N)',fontsize=12)
    cbar.set_alpha(1)
    cbar.draw_all() 
    target = os.path.join(root, 'models',model_ID, 'Results','Hourly', 'Figures', fname + '.png')      
    plt.savefig(target,dpi=150)
    plt.close("all")

def combine_maps_make_figures(date):
    for hr in range(24):
        filenames = ROI+'_'+modeln+r'_no2_'+date.strftime('%Y-%m-%d')+'T' +str(hr).zfill(2)+'_*.nc'   
        fname = filenames.split('*')[0][:-1]
        newname = fname+'.nc'
        target = os.path.join(root, 'models',model_ID, 'Results', 'Hourly', newname)            
        if not os.path.isfile(target): 
            listfns = glob.glob(os.path.join(root, 'models',model_ID, 'Results',filenames))      
            if len(listfns) == 36:
                try:
                    with xr.open_mfdataset(listfns, combine='by_coords') as ds:
                        ds.to_netcdf(target)
                        plot_save_fig(ds, fname)
                except:
                    print('file not saved:' + fname)                       
            else:
                print(len(listfns))
                print('Data missing:'+fname) 
        else:
            print('file exists:' + fname) 
            
def load_maps_make_figures(date):
    for hr in range(24):
        filename = ROI+'_'+modeln+r'_no2_'+date.strftime('%Y-%m-%d')+'T' +str(hr).zfill(2)+'.nc'   
        target = os.path.join(root, 'models',model_ID, 'Results', 'Hourly', filename)             
        with xr.open_dataset(target) as ds:
            plot_save_fig(ds, filename[:-3])


# %%
def main():
    
    description = textwrap.dedent("""\
        make predictions of NO2 map with the best model using XGB (should be already saved and named as 'XGB_best_'+modeln+'.sav' )
        Here, there are several models (named 'modeln') using different features
        ---------------------------------------        
        Model1: Null model : Using data without extraction of particular features as other models
        Model2. Spatial features : Feature extraction of emission sources (max values and max distance in range [0,100])
        Model3. Spatial features + wavelet transform of DEM (digital elevation model) 
        Model4. dimensionality reduction  (rebate), remove some features 
        -------------------------------------
        
        This code requires 4 input arguments;
        arg1: Model name
        arg2: start date to plot (e.g., 2019-03-01) 
        arg3: stop date to plot (e.g., 2019-04-01)
        arg4: qa threshold values for training set (50 or 75)
        
        As a result, there will subsections of predicted maps (netCDF files) of NO2 will be saved in the folder    
        os.path.join(root, 'models',model_ID, 'Results')
        with the name: ROI+'_'+modeln+r'_no2_'+str(t_obs.values)[:13]+'_lon_%5f_lat_%5f.nc' 
        36 netcdf files can be combiled to a single netcdf file for each time (hour)
        by using the script 'merge_predicted_ROI_maps.py'
    
        by Minsu Kim (minsu.kim@empa.ch)
        ORCID:https://orcid.org/0000-0002-3942-3743
        
    """)
        
    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('Model', type=str, help='model name (Model1, Model2, Model3, Model4)')
    parser.add_argument('starttime', type=str, help='start date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('stoptime', type=str, help='stop date (YYYY-mm-dd or YYYY-jjj)')   
    parser.add_argument('qa', type=int, help='where to cut the training data; qualitiy assurance level of 50 or75')

    global root, ROI,model_ID, modeln, newcmp, rgb

    args = parser.parse_args()
    modeln = args.Model
    qa_val = args.qa

    root = '.'
    ROI = 'ROI1'
    model_ID = 'Full_features'+str(qa_val) 
    
    cm0 = LinearSegmentedColormap.from_list(
            'cm0', ['midnightblue','darkturquoise',(1,1,0.5)], N=64)                                   
    cm1 = LinearSegmentedColormap.from_list(
            'cm1', [(1,1,0.5),'darkorange','firebrick','maroon'], N=192)  
    newcolors = np.vstack((cm0(np.linspace(0, 1, 128)),
                           cm1(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors)
    
    ls = LightSource(azdeg=315, altdeg=45)
    # Spatial co-variate maps are already prepraeed in a sinlge netCDF.
    # Here the data is called for topography (digital elevapation map) for plotting reasons
    ds2 = xr.open_dataset(os.path.join(root, 'input','ROI1_v2.nc'))
    elev = ds2.dem.data
    ve = 0.3
    rgb = ls.hillshade(elev, vert_exag=ve,dx = 20, dy = 20)
    
    start = parse_date(args.starttime)
    stop = parse_date(args.stoptime)
    
    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(combine_maps_make_figures)(date) for date in iter_dates(start, stop))
    
    
if __name__ == '__main__':
    main()

