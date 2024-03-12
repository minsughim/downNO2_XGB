# -*- coding: utf-8 -*-
"""
This code extract the road information of region of interest and process per nuts3 region
at the end of the code each nut3 regions are combined in a single raster file

by Minsu Kim (minsu.kim@empa.ch)
"""
# TODO : better documenting!!
# TODO : clear path dependency 

import os
import pandas as pd
import shapefile
from shapely.geometry import MultiLineString, Polygon, box
import numpy as np
import xarray as xr
#from distributed import Client
#client = Client() # For local 
#client = Client(scheduler_file=os.environ['DASK_SCHED_FILE']) #For Piz Daint
import glob

# %% to make a target raster file using the same resolution of land usage data :: Should be done once for rasterising
#filenc = './input/LUD/g100_clc18_ROI2.nc'
#dsROI = xr.open_dataset(filenc)
#lat = dsROI.lat
#lon = dsROI.lon
#devlon = (lon -np.roll(lon,1))*0.5
#devlon[0] = devlon[1]
#devlat = (lat - np.roll(lat,1))*0.5
#devlat[0] = devlat[1]
#lonv, latv = np.meshgrid(lon,lat)
#lond, latd = np.meshgrid(devlon,devlat)
#ds2 = xr.Dataset({'lonvV': (['lat', 'lon'],  lonv),
#                 'latvV': (['lat', 'lon'],  latv),
#                 'lonvD': (['lat', 'lon'],  lond),
#                 'latvD': (['lat', 'lon'],  latd)},
#                 coords={'lon':lon,'lat':lat})
#ds2['linelength'] = xr.zeros_like(ds2.lonvV)
#ds2['trafficvolume'] = xr.zeros_like(ds2.lonvV)
#ds2.to_netcdf(r'/scratch/snx3000/minsukim/input/OSMtraffic/spatial_coords_ROI2.nc')

# %% Functions
def make_grid_cells(pos):
    polygon = [(pos[0]-pos[2], pos[1]-pos[3]), (pos[0]+pos[2], pos[1]-pos[3]),(pos[0]+pos[2], pos[1]+pos[3]),(pos[0]-pos[2], pos[1]+pos[3]),(pos[0]-pos[2], pos[1]-pos[3])]
    shapely_poly = Polygon(polygon)
    return shapely_poly

def extract_traffic_info(i):   
    target = PATH+r'/'+ROI+r'/roadlinks_'+i+'.nc'
    if not os.path.isfile(target): 
        shp_path = PATH+r'/traffic/'+i+r'/roadlinks_'+i+'.shp'
        dbf_path = PATH+r'/traffic/'+i+r'/roadlinks_'+i+'.dbf'
        myshp = open(shp_path, 'rb')
        mydbf = open(dbf_path, 'rb')
        r = shapefile.Reader(shp=myshp, dbf=mydbf, encoding='latin-1')
        rbox = box(r.bbox[0],r.bbox[1],r.bbox[2],r.bbox[3])
        if rbox.intersects(ROIbox): # Check whether the area is in the region of interest      
            try:
                spatial_coords = xr.open_dataset(r'/scratch/snx3000/minsukim/input/OSMtraffic/spatial_coords_'+ROI+r'.nc')
                nut3region = spatial_coords.sel(lon=slice(r.bbox[0],r.bbox[2]),lat=slice(r.bbox[1],r.bbox[3]))      
                roadlist = r.shapes()
                roadinfo = r.records()
                for info, roadname in zip(roadinfo, roadlist):            
                    roadlines = MultiLineString([roadname.points])
                    roadgrids = nut3region.sel(lon=slice(roadlines.bounds[0]-thres,roadlines.bounds[2]+thres),lat=slice(roadlines.bounds[1]-thres,roadlines.bounds[3]+thres))
                    grid_cells = np.stack([roadgrids.lonvV.values.ravel(), 
                                          roadgrids.latvV.values.ravel(),
                                          roadgrids.lonvD.values.ravel(),
                                          roadgrids.latvD.values.ravel()], axis=1) 
                    totLength = roadlines.length
                    
                    try: 
                        tvol = info['trafficvol']                                    
                        for pos in grid_cells:
                            grid_cell =  make_grid_cells(pos)
                            intersectlength = roadlines.intersection(grid_cell).length
                            nut3region.linelength.loc[pos[1],pos[0]] += intersectlength
                            try:
                                nut3region.trafficvolume.loc[pos[1],pos[0]] += tvol*intersectlength/totLength
                            except:
                                pass
                    except:
                        print(i)
                        print('no traffic volume')  
                        tvol = -1
                        for pos in grid_cells:
                            grid_cell =  make_grid_cells(pos)
                            intersectlength = roadlines.intersection(grid_cell).length
                            nut3region.linelength.loc[pos[1],pos[0]] += intersectlength
                            try:
                                nut3region.trafficvolume.loc[pos[1],pos[0]] += tvol*intersectlength/totLength
                            except:
                                pass
                        
                nut3region = nut3region.chunk(chunks={'lon':256, 'lat':256})
                target = PATH+r'/'+ROI+r'/roadlinks_'+i+'.nc'
                nut3region.to_netcdf(target)
            except:
                print(i)
                print('no traffic data processed')                
    return None

# %%  assigning global variables, PATH and region of interests 


global PATH, ROIbox, thres, ROI

ROI = 'ROI2'
PATH = r'/scratch/snx3000/minsukim/data/OSMtraffic' #for piz daint
dataSpatial = xr.open_dataset(r'/scratch/snx3000/minsukim/input/'+ROI+ r'.nc') #use the land usage data (100 m resolution) as the targeted raster file

#PATH = r'./data/traffic' #for local
#dataSpatial = xr.open_dataset(r'./data/ROI1.nc')
lat = dataSpatial.lat   
lon = dataSpatial.lon
devlon = (lon -np.roll(lon,1))*0.5
devlon[0] = devlon[1]
devlat = (lat - np.roll(lat,1))*0.5
devlat[0] = devlat[1]
thres = 0.001 #threshold for selecting a pixel for a road defined as a pixel resolution
ROIbox = box(lon[0]-devlon[0],lat[0]-devlat[0],lon[-1]+devlon[-1],lat[-1]+devlat[-1])
#params = client.scatter({'PATH':PATH,'ROIbox':ROIbox,'thres':thres, 'ROI':ROI}, broadcast=True)

# %% extract traffic and road information in csv file for each nuts3
df = pd.read_excel(r'/scratch/snx3000/minsukim/input/OSMtraffic/nuts3.xls')  #for local
ids = df['NUTS 3 ID (2010)'].dropna().values
from joblib import Parallel, delayed
Parallel(n_jobs=-1)(delayed(extract_traffic_info)(idn) for idn in ids)
#
#import joblib
#with joblib.parallel_backend('dask'): 
#    joblib.Parallel(verbose=100)(joblib.delayed(extract_traffic_info)(idn) for idn in ids)

#futures = client.map(extract_traffic_info, ids) # assign jobs using dask clients
#save_files = client.gather(futures)


# %% Concat results to create one raster file

filename = r'/scratch/snx3000/minsukim/input/OSMtraffic/spatial_coords_'+ROI+r'.nc'
raster_traffic = xr.open_dataset(filename, drop_variables={'lonvV','latvV', 'lonvD','latvD'})
#raster_traffic = xr.open_dataset(r'./input/OSMtraffic/spatial_coords.nc', drop_variables={'lonvV','latvV', 'lonvD','latvD'})
raster_traffic.linelength[:] = np.nan
raster_traffic.trafficvolume[:] = np.nan
length = np.full_like(raster_traffic.linelength,np.nan)
volume = np.full_like(raster_traffic.linelength,np.nan)

pathname = PATH+r'/'+ROI+r'/roadlinks_*.nc'
filenames = glob.glob(pathname)

for f in filenames:
    ds = xr.open_dataset(f, drop_variables={'lonvV','latvV', 'lonvD','latvD'})
    _,dsa = xr.align(raster_traffic,ds,join='left')
    bdsa = (dsa.linelength.values > 0)
    length[bdsa] = dsa.linelength.values[bdsa]
    volume[bdsa] = dsa.trafficvolume.values[bdsa]
    print(f)

raster_traffic.linelength[:] = length
raster_traffic.trafficvolume[:] = volume

target = PATH+r'/'+ROI+r'/traffic_raster.nc'
raster_traffic.to_netcdf(target)


