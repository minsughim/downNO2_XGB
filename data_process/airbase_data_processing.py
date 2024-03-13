# -*- coding: utf-8 -*-
"""  
Python script for processing AIRBASE data for machine learning algorithms,
This script processes downloaded AIRBASE data using airbase_data_down.py(/downno2/data_download/airbase_data_down.py). 
In this script sorts stations of NO2 and sort the staion information and the timeseries in a folder of the respective region of interest (ROI1, Alpine, and ROI2, Benelux)
The meta information file of AIRBASE can be downloaded from https://polybox.ethz.ch/index.php/s/hTyhS24aDjjqkdP
For detailed information, https://www.eea.europa.eu/data-and-maps/data/aqereporting-8#tab-metadata 

Reference: 
Minsu Kim, Dominik Brunner, Gerrit Kuhlmann (2021) 
Importance of satellite observations for high-resolution mapping of near-surface NO2 by machine learning, 
Remote sensing of Environment DOI: https://doi.org/10.1016/j.rse.2021.112573

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743
"""
import pandas as pd
import os
import glob

PATH ='./'
file = os.path.join(PATH, 'input', 'AIRBASE', 'metadata_AIRBASE.csv')
metaBase = pd.read_csv(file)
df = pd.DataFrame(metaBase)
no2_airbase = df.groupby('AirPollutant').get_group('NO2')


# %% Define region of interests (ROI1: Alpine region, ROI2: Benelux)
bROI1 = (df.Longitude>6)&(df.Longitude<12)&(df.Latitude>42)&(df.Latitude<48)
bROI2 = (df.Longitude>2)&(df.Longitude<8)&(df.Latitude>48)&(df.Latitude<54)
no2_ROI1 = no2_airbase[bROI1]
no2_ROI2 = no2_airbase[bROI2]

ROIlist = ['no2_ROI1','no2_ROI2']
# %% processing downloaded airbase data base on regions of interest

datafiles = os.path.join(PATH, 'data', 'AIRBASE', '*2018_timeseries.csv')
filenames = glob.glob(datafiles)

for f in filenames:
    
    data = pd.read_csv(f,usecols=[4,10,11,12,13,14,15,16],keep_date_col=True,date_parser=True,encoding='utf-16')
    stnname = data['AirQualityStationEoICode'][1]
    filename = os.path.basename(f)
# %TODO: Some files in airbase data occur encoding error. needs to be fixed to automate the entire process
# by abling these lines, compiled files include data from before 2018 if exsits 
    stnfilenames = './data/AIRBASE/'+filename[:10]+'*.csv'          
    stnfiles = glob.glob(stnfilenames)
    stnfiles.sort() 
    if len(stnfiles) == 2:
        df1 = pd.read_csv(stnfiles[0],usecols=[4,10,11,12,13,14,15,16],keep_date_col=True,date_parser=True,encoding='utf-16')
        df2 = pd.read_csv(stnfiles[1],usecols=[4,10,11,12,13,14,15,16],keep_date_col=True,date_parser=True)
        df = pd.concat( [df1, df2]) 
        newfilename = stnname+'_'+'2018_2019.csv'
        for ROI in ROIlist:
            if (eval(ROI).AirQualityStationEoICode == stnname).sum():
                print(ROI)
                target = os.path.join(PATH, 'data', 'AIRBASE', ROI ,newfilename)
                df.to_csv(target)
            else:
                print('elsewhere')
                target = os.path.join(PATH, 'data', 'AIRBASE', 'elsewhere' ,newfilename)
                df.to_csv(target)       
    else:
        print(stnname+'::number of files:'+str(len(stnfiles)))
          
