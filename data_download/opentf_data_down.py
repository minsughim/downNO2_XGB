# -*- coding: utf-8 -*-
"""
Python script for downloading traffic data from opentrasportmap
For details, http://opentransportmap.info/

The input file (nuts3.xls) is the list of nuts3 from EUROSTATS
The file can be downloaded from https://polybox.ethz.ch/index.php/s/UnuQNtrDS0DiDVw

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""

import pandas as pd
import urllib.request
import zipfile
import os

#%% main function

def main():

    PATH = r'./data/traffic'
    url = r'http://opentransportmap.info/download/nuts-3/'
    df = pd.read_excel(r'./data/traffic/nuts3.xls') #package xlrd
    ids = df['NUTS 3 ID (2010)'].dropna().values

    for i in ids:
	
        print(r'downloading {:}...'.format(i))
        link = url+i
        try:
            urllib.request.urlretrieve(link, PATH+r'/'+i+'.zip')
            zip_ref = zipfile.ZipFile(PATH+r'/'+i+'.zip', 'r')
            zip_ref.extractall(PATH+r'/'+i)
            zip_ref.close()
            os.remove(PATH+r'/'+i+'.zip')
            print('done')
        except:
            print('no file found')
        
if __name__ == '__main__':
    main()
       
