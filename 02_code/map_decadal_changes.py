# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:38:04 2023

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
parser.add_argument('years', help='provide comma-separated list \
                    of years corresponding to each input file', 
    type=lambda s: [int(item) for item in s.split(',')])

options = parser.parse_args()
config = options.config
files = options.files
years = options.years

# extract base directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))
cdir = join(wdir, config['change_dir'])

#----------------------------------------------------------------------------#
# identify pixels for which to derive predictions
#----------------------------------------------------------------------------#

iname = join(wdir, '00_data', 'pixelArea-hectares_20190000_300m.tif')
reference = rt.open(iname)

# load tiling shapefile
fid = ('{0:0' + str(len(str(len(sp)))) + 'd}').format(index)
bb = shape(sp[index]['geometry']).bounds
# extract pixel resolution (used to infer start/end pixel coordinates)
pr = reference.res[0]

# define input window
sp = reference.index(bb[0]+pr/2,bb[3]-pr/2)
nr = int(np.round((bb[3]-bb[1])/pr))
nc = int(np.round((bb[2]-bb[0])/pr))
w = rt.windows.Window(sp[1],sp[0],nc,nr)

# define start x/y coordinates of outputs
ot = rt.transform.from_origin(bb[0],bb[3],pr,pr)

# build metadata profile of outputs
p = reference.meta.copy()
p.update(driver='GTiff', dtype='uint16', 
          compress='deflate', nodata=None, 
         predict=2, zlevel=9, height=nc, 
         width=nc, transform=ot)

#----------------------------------------------------------------------------#
# extract data on directories
#----------------------------------------------------------------------------#

ia = np.zeros((nr,nc,nf), dtype='float32')
area = reference.read(1, window=w)

for f in range(0,nf):
    a = rt.open(files[f]).read(1, window=w).astype('float32')
    a[a > 100] = 0
    ia[:,:,f] = a*0.01*area
    print(f)

ia = np.abs(np.diff(ia, axis=2))
years = years[1:len(files)]

oa = np.zeros((nr,nc,3), dtype="float32")
time_ranges = [[1992,1999],[2000,2010],[2011,2018]]
for r in range(0,len(time_ranges)):
    tr = time_ranges[r]
    si = np.where(np.isin(years, range(tr[0],tr[1]+1)))[0]
    oa[:,:,r] = np.sum(ia[:,:,si], axis=2)

mx = np.max(oa, axis=2)
lr = np.argmax(oa, axis=2)
lr[np.where(mx > 0)] += 1

#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#

p.update(dtype='float32')
ods = rt.open(join(wdir, 'tmp', 'changeStats', 
                   f'{fid}_{variable}_{yrange}_maxChange.tif'), 'w', **p)
ods.write(mx.astype("float32"), indexes=1)
ods.close()

p.update(dtype='uint8')
ods = rt.open(join(wdir, 'tmp', 'changeStats', 
                   f'{fid}_{variable}_{yrange}_maxDecade.tif'), 'w', **p)
ods.write(lr.astype("uint8"), indexes=1)
ods.close()
