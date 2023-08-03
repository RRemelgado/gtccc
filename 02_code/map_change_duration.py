# -*- coding: utf-8 -*-
""" map change duration
------------------------------------------------------------------------------
Characterization of the duraion of changes in tree-canopy cover over 
pixels with significant losses/gains. For each pixel in a given list 
of files, the algorithm will calculate year-to-year differences. Then, 
if dealing with pixels classified as "losses", the algorithm will then 
isolate the years with negative year-to-year differences, sort them, and 
calculate the amount of years required to achieve at least 80% of the 
balance between the first and the total change. When dealing with pixels 
classified as "gain", the algorithm instead isolates pixels with positive 
year-to-year differences. It is assumed that the change classification 
layer distinguishes "losses" with a grid code of 3, and "gains" with a 
grid code of 4, as proided by the script "map_change_types.py".

------------------------------------------------------------------------------
------------------------------------------------------------------------------
Created on Mon Feb 27 11:18:59 2023
@author: Ruben Remelgado
"""

#----------------------------------------------------------------------------#
# load modules/arguments
#----------------------------------------------------------------------------#

from argparse import ArgumentParser
from os.path import join, dirname, basename
import rasterio as rt
import numpy as np
import yaml

parser = ArgumentParser(description = 'classify tropical biomes')

parser.add_argument('files', help='provide \
                    comma-separated list of files to analyse', 
    type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument("change", help="layer with per-pixel \
                    classification of changes", type="str")

options = parser.parse_args()
config = options.config
files = options.files
change = options.change
nf = len(files)

# extract base directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))
cdir = join(wdir, config['change_dir'])

#----------------------------------------------------------------------------#
# locate first image in list to retrieve metadata
#----------------------------------------------------------------------------#

change = rt.open(change)
nr = change.height
nc = change.width

p = change.meta.copy()
p.update(dtype='uint8', compress='deflate', predict=2, zlevel=9, nodata=255)

#----------------------------------------------------------------------------#
# load time-series on tree-canopy cover
#----------------------------------------------------------------------------#

ia = np.zeros((nr,nc,nf), dtype='float32')

for f in range(0,nf):
    ids = rt.open(files[f])
    a = ids.read(1)
    ia[:,:,f] = a
    ids.close()
    print(f)

#----------------------------------------------------------------------------#
# calculate number of years required to achieve 80% of the total loss
#----------------------------------------------------------------------------#

# load data on classified changes
change = rt.open(change).read(1)

nr_years = np.zeros((nr,nc), dtype='uint8')

# find pixels with losses/gains
px = np.where(change == 3)

# iterate through each pixel with losses
for i in range(0,len(px[0])):
    
    # extract data for pixel and calculate year-to-year differences
    v = np.append([0],np.diff(ia[px[0][i],px[1][i],:].copy()))
    
    # get absolute of losses
    s = np.abs(v[v < 0])
    
    # count out many years were required to reach 80% of the time-series loss
    nr_years[px[0][i],px[1][i]] = \
        np.where(np.cumsum(s[np.argsort(s)] / np.sum(s)) >= 0.8)[0][0]+1
    
    print(i)

bname = basename(files[0]).split(".")[0]
ods = rt.open(join(f'{bname}_nrYears80pLoss.tif'), 'w', **p)
ods.write(nr_years, indexes=1)
ods.close()

#----------------------------------------------------------------------------#
# calculate number of years required to achieve 80% of the total gain
#----------------------------------------------------------------------------#

# find pixels with losses/gains
px = np.where(change == 3)
nr_years[:] = 0

# iterate through each pixel with losses
for i in range(0,len(px[0])):
    
    # extract data for pixel and calculate year-to-year differences
    v = np.append([0],np.diff(ia[px[0][i],px[1][i],:].copy()))
    
    # get absolute of losses
    s = np.abs(v[v > 0])
    
    # count out many years were required to reach 80% of the time-series loss
    nr_years[px[0][i],px[1][i]] = \
        np.where(np.cumsum(s[np.argsort(s)] / np.sum(s)) >= 0.8)[0][0]+1
    
    print(i)


ods = rt.open(join(f'{bname}_nrYears80pGain.tif'), 'w', **p)
ods.write(nr_years, indexes=1)
ods.close()

#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
