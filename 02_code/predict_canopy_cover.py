# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:29:02 2022
@author: Ruben Remelgado
"""

#----------------------------------------------------------------------------#
# load arguments and modules
#----------------------------------------------------------------------------#

from argparse import ArgumentParser
from shapely.geometry import shape
from os.path import join, dirname
import rasterio as rt
import pandas as pd
import fiona as fn
import numpy as np
import pickle
import yaml

parser = ArgumentParser(description = 'Classification \
                        of canopy cover densities')
parser.add_argument('config', help='configuration file', type=str)
parser.add_argument('index', help='index of target tile', type=int)
parser.add_argument('year', help='year for which to map', type=str)
parser.add_argument('gap', help='temporal gap between \
                    target and reference years', type=int)

options = parser.parse_args()
config = options.config
index = options.index
year = options.year
gap = options.gap

# extract base directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))

# access geocomputation boundaries
sp = fn.open(join(wdir, config['tiles']))

# define unique identifier for the tile
tid = ('{0:0' + len(str(len(sp))) + 'd}').format(index)

# extract bounding geometry from tile (used to subset input data)
bb = shape(sp[index]['geometry']).bounds

# define output directories
cdir = join(wdir, config['canopy_dir'])
udir = join(wdir, config['uncertainty_dir'])

#----------------------------------------------------------------------------#
# locate  pixels for which to derive predictions
#----------------------------------------------------------------------------#

land_mask = rt.open(join(wdir, config['mapping_mask']))
pr = land_mask.res[0]

sp = land_mask.index(bb[0]+pr/2,bb[3]-pr/2)
nr = int(np.round((bb[3]-bb[1])/pr))
nc = int(np.round((bb[2]-bb[0])/pr))
w = rt.windows.Window(sp[1],sp[0],nc,nr)
ot = rt.transform.from_origin(bb[0],bb[3],pr,pr)

mask = land_mask.read(1,window=w)

p = land_mask.meta.copy()
p.update(dtype='float32', compress='deflate', predict=2, 
         zlevel=9, height=nc, width=nc, transform=ot, nodata=255)

px = np.where(mask == 1)
del mask

#----------------------------------------------------------------------------#
# load predictors
#----------------------------------------------------------------------------#

# list variables and their paths
variables = list(config['predictors'].keys())

ids = []

# extract data
for var in variables:
    ids.append(rt.open(join(wdir, 
                            year.join(config['predictors'][var].split('*')))))

# combine data into data frame
predictors = np.zeros((len(px[0]),len(variables)), dtype='float32')
for v in range(0,len(ids)):
    predictors[:,v] = ids[v].read(window=w, indexes=1)[0][px]
predictors = pd.DataFrame(predictors, columns=variables)
predictors['gap'] = int(gap)

#----------------------------------------------------------------------------#
# predict and derive uncertainties
#----------------------------------------------------------------------------#

# load/apply model
model = pickle.load(open(join(wdir, config['model']['path']), 'rb'))
prediction = model.predict(predictors)

# save prediction
oa = np.zeros((nr,nc), dtype='float32')
oa[:] = 255
oa[px] = prediction
ods = rt.open(join(cdir, 'tmp', f'{tid}_{year}_tcc.tif'), 'w', **p)
ods.write(oa, indexes=1)
ods.close()

# estimate uncertainties
uncertainty = np.zeros(len(px[0]), dtype='float32')
for e in range(0,len(model.estimators_)):
    uncertainty += (prediction-model.estimators_[e].predict(predictors))**2
    print(e)
si = np.where(uncertainty > 0)
uncertainty[si] = np.sqrt(uncertainty[si]/ len(model.estimators_))

del predictors, prediction, 

# save uncertainties
oa[px] = uncertainty
ods = rt.open(join(udir, f'{tid}_{year}_tcc_ci.tif'), 'w', **p)
ods.write(oa.astype("float32"), indexes=1)
ods.close()
del uncertainty

#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
