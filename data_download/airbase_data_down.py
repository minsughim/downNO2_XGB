# -*- coding: utf-8 -*-
"""
Python script for downloading NO2 time series from AIRBASE data
For details, http://discomap.eea.europa.eu/map/fme/AirQualityExport.htm
The input file (Airbase_links.txt) of list of links to dowload all NO2 time series available in AIRBASE dataset
@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743
"""
import pandas as pd
import os

# %% main function

def main():

    PATH = '.'
  
    filename = r'./input/AIRBASE/Airbase_links.txt'
    linklist =pd.read_csv(filename, header=None, dtype=str)

    for link in linklist.loc[:,0]:
        
        if not os.path.isfile(os.path.join(PATH,'data','Airbase', link.split('/')[-1])):

            year = link[-19:-15]
            target = os.path.join(PATH, 'data','Airbase',year)
            print(link)
            # download
            command = ' '.join([
                    'wget', '-q', '-r', '-l1', '-nd', '-nc', '--no-parent',
                    '--directory-prefix=%s' % os.path.dirname(target), link
                    ])
            os.system(command)


if __name__ == '__main__':
    main()



