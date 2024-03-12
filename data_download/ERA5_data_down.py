# -*- coding: utf-8 -*-
"""
Python script for downloading ERA5 data (https://cds.climate.copernicus.eu/api-how-to)
Here, 5 variables are selected
'10m_u_component_of_wind',
'10m_v_component_of_wind',
'2m_temperature',
'clear_sky_direct_solar_radiation_at_surface',
'total_precipitation'

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""
import cdsapi # necessary api for downloading ERA5
from datetime import datetime, timedelta
import argparse
import textwrap
import os

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

def download(start, stop, root='.'):
        
    for date in iter_dates(start, stop):

        filename = 'ERA5_ROI_%Y%m%d.nc'
        print(date.strftime(filename))
        target = os.path.join(root, 'ERA5',date.strftime(filename))

        c = cdsapi.Client()
        
        c.retrieve(
                'reanalysis-era5-single-levels',
                {
                        'product_type':'reanalysis',
                        'variable':[
                                '10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature',
                                'clear_sky_direct_solar_radiation_at_surface','total_precipitation'
                                    ],
                        'year': date.strftime('%Y'),
                        'month': date.strftime('%m'),
                        'day': date.strftime('%d'),
                        #'area':[48, 6, 42, 12], #North, West, South, East (ROI1)
                        'area':[54, 2, 48, 8], #North, West, South, East (ROI2) [latmax, lonmin, latmin, lonmax]
                        'time':[
                                '00:00','01:00','02:00',
                                '03:00','04:00','05:00',
                                '06:00','07:00','08:00',
                                '09:00','10:00','11:00',
                                '12:00','13:00','14:00',
                                '15:00','16:00','17:00',
                                '18:00','19:00','20:00',
                                '21:00','22:00','23:00'
                                ],
                        'format':'netcdf'
                },
                target)

# %% main function

def main():
    """
    Paramters
    ------
    starttime : str
        start date (YYYY-mm-dd or YYYY-jjj)

    stoptime : str
        stop date (YYYY-mm-dd or YYYY-jjj)

    PATH 

    returns
    ------
    ERA5 data saved in netcdf 
    PATH = PATH/ERA5/
    filename = 'ERA5_ROI_%Y%m%d.nc'

    """

    PATH = './data/'

    description = textwrap.dedent("""\   
        Python script for downloading ERA5 reanalysis data hourly averaged dialy data at global scale:
        For details, https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
        
        This script is modified from a script, domino.py by Gerrit Kuhlmann
        (https://gitlab.empa.ch/abt503/users/kug/satdownload/blob/master/domino.py)
        
        by Minsu Kim (minsu.kim@empa.ch)
    """)
        
    parser = argparse.ArgumentParser(description=description, epilog='',
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('starttime', type=str, help='start date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('stoptime', type=str, help='stop date (YYYY-mm-dd or YYYY-jjj)')
    parser.add_argument('--prefix', default='.', type=str, help='PATH')

        
    args = parser.parse_args()
    
    start = parse_date(args.starttime)
    stop = parse_date(args.stoptime)
    
    download(start, stop, root=PATH)

if __name__ == '__main__':
    main()




