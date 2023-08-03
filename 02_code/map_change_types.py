# -*- coding: utf-8 -*-
""" map change types
------------------------------------------------------------------------------
Classification of changes in tree-canopy cover. The algorithm loads 
a list of files with the same spatial resolution and extent and, for 
each pixel, applies the Mann-Kendall test to infer the significance 
of Long-term trend. If the reported significance is below a user-defined 
threshold, the algorithm classifies "losses" (class code 3) when the slope 
of the time-series is negative, and "gains" (class code 4) when the slope 
is positive. When the significance is above the specified threshold and 
changes are present, wedescribe "disturbances" (class code 5). If changes 
are not present, we distinguish between "tree-covered" (when there is no 
change and tree cover, class code 2) and "not tree covered" (when there 
is no change, and no tree cover).
------------------------------------------------------------------------------
------------------------------------------------------------------------------
Created on Fri Aug  5 12:56:47 2022
@author: Ruben Remelgado
"""

#----------------------------------------------------------------------------#
# load modules/arguments
#----------------------------------------------------------------------------#

import pymannkendall as mk
from argparse import ArgumentParser
from os.path import join, dirname, basename
import rasterio as rt
import numpy as np
import yaml

parser = ArgumentParser(description = 'canopy-cover change classification')
parser.add_argument('config', help='configuration file', type=str)
parser.add_argument('files', help='provide \
                    comma-separated list of files to analyse', 
    type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('pval', help='significance threshold', type=float)

options = parser.parse_args()
config = options.config
files = options.files
pval = options.pval
nf = len(files)

# extract base directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))
cdir = join(wdir, config['change_dir'])

#----------------------------------------------------------------------------#
# locate first image in list to retrieve metadata
#----------------------------------------------------------------------------#

reference = rt.open(files[0])
pr = reference.res[0]
nr = reference.height
nc = reference.width

p = reference.meta.copy()
p.update(dtype='uint8', compress='deflate', predict=2, zlevel=9, nodata=255)

#----------------------------------------------------------------------------#
# load data
#----------------------------------------------------------------------------#

ia = np.zeros((nr,nc,nf), dtype='float32')
ca = np.zeros((nr,nc), dtype='float32')

for f in range(0,nf):
    ids = rt.open(files[f])
    a = ids.read(1)
    a[a > 100] = 0
    ia[:,:,f] = a
    ids.close()
    
    # preserve initial step
    if f == 0:
        b = a.copy()
    
    # find if new step is different from the previous
    if f > 0:
        ca[a != b] += 1
        b = a.copy()
    
    print(f)

#----------------------------------------------------------------------------#
# evluate changes
#----------------------------------------------------------------------------#

min_cover = np.min(ia, axis=2)
max_cover = np.max(ia, axis=2)

slope = np.zeros((nr,nc), dtype='float32')
slope[:] = 999
pvalue = np.zeros((nr,nc), dtype='float32')
pvalue[:] = 999

px = np.where((min_cover < max_cover) & (ca > 1))

# account for pixels with multiple changes
for i in range(0,len(px[0])):
    t = mk.original_test(np.reshape(ia[px[0][i],px[1][i],:],(nf,1)))
    slope[px[0][i],px[1][i]] = t.slope
    pvalue[px[0][i],px[1][i]] = t.p
    print(i)

# account for clear breaks in time-series
px = np.where((min_cover < max_cover) & (ca == 1))
slope[px] = 1
pvalue[px] = 1
px = np.where((min_cover < max_cover) & (ca == 1))
slope[px] = -1
pvalue[px] = 1

del ia

#----------------------------------------------------------------------------#
# classify abd save changes
#----------------------------------------------------------------------------#

oa = np.zeros((nr,nc), dtype='uint8')
oa[(min_cover == max_cover) & (min_cover == 0)] = 1 # stable non-forested
oa[(min_cover == max_cover) & (min_cover > 0)] = 2 # stable forested
oa[(pvalue < pval) & (slope < 0)] = 3 # loss
oa[(pvalue < pval) & (slope > 0)] = 4 # gain
oa[((pvalue >= pval) | (slope == 0)) & (oa == 0)] = 5 # unstable

bname = basename(files[0]).split(".")[0]
ods = rt.open(join(cdir, f'{bname}_changeType.tif'), "w", **p)
ods.write(oa, indexes=1)
ods.close()

#----------------------------------------------------------------------------#
# write output
#----------------------------------------------------------------------------#

p.update(dtype='float32', nodata=999)
ods = rt.open(join(cdir, f'{bname}_changeSlope.tif'), "w", **p)
ods.write(slope.astype("float32"), indexes=1)
ods.close()

ods = rt.open(join(cdir, f'{bname}_changePvalue.tif'), "w", **p)
ods.write(pvalue.astype("float32"), indexes=1)
ods.close()
