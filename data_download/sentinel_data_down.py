# -*- coding: utf-8 -*-
"""
Python script for downloading Sentinel 5p data (Level 2) from 'http://www.temis.nl/airpollution/'

@author: Minsu Kim (minsu.kim@empa.ch) at Empa - Swiss Federal Laboratories for Materials Science and Technology
ORCID:https://orcid.org/0000-0002-3942-3743

"""

from datetime import datetime, timedelta
import argparse
import os
import textwrap

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
    
    filename = 'tropomi_no2_%Y%m%d.tar'
    
    for date in iter_dates(start, stop):
    
        link = '/'.join([
            'http://www.temis.nl/airpollution/no2col/data/tropomi',
            date.strftime('%Y'), date.strftime('%m'), date.strftime(filename)
        ])
        target = os.path.join(root, 'Sentinelno2',
                              date.strftime('%Y'),
                              date.strftime(filename))
    
        print(date.strftime('%Y-%m-%d'))
    
        # download
        command = ' '.join([
            'wget', '-q', '-r', '-l1', '-nd', '-nc', '--no-parent',
            '--directory-prefix=%s' % os.path.dirname(target), link
        ])
        os.system(command)
 
        # untar
        os.system('tar -xf %s -C %s' % (target, os.path.dirname(target)))
    
        # remove target
        os.remove(target)
        
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
    TROPOMI data saved in netcdf

    """

    PATH = './data/'
    
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
    parser.add_argument('--prefix', default='.', type=str, help='PATH folder')
    
    args = parser.parse_args()
    
    start = parse_date(args.starttime)
    stop = parse_date(args.stoptime)
    
    download(start, stop, root=PATH)

if __name__ == '__main__':
    main()

