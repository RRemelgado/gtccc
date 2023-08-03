# -*- coding: utf-8 -*-
"""  sample
------------------------------------------------------------------------------
This algorithm selects samples to support the prediciton of tree-canopy 
cover First, the algorithm estimates the number of samples to be extracted. 
This number is estimated from available area with valid pixels within the 
land mask and without forest cover gains, assuring the retrieval of at least 
1 sample per each 50 km2. This number is then divided by 2, as to sample:
    
    a) One set of samples collected from pixels with forest changes
    b) One set of samples collected from pixels without forest changes
    
For each sample groups, the target number is again subvided for each land 
cover class. The number of samples assigned to each class is proportional 
to the respective proportion of land. The forest change layer on which the 
samplign will be conducted is estimated itnernally, provided two layers: one 
for the starting forest cover, and one for the ending forest cover.

When iteracting through a land cover class, the two sampling groups receive 
different treatments. In the stable group, we use a random sampling. In the 
changing group, we use right-skewed sampling. First, the algorithm sorts 
pixels by their magnitude of change. Then, if generates a random, right-skewed 
distribution with the same length as the indices of pixels with forest change. 
This distribution is used to select observations from the sorted indices, 
assuring  the number of samples across different magnitudes of change are 
balance. Because large changes are naturally rarer than subtle ones, a simple 
random sampling will bias the sample distriution towards lower magnitudes of 
change.

After iterating through every land cover class, the algorithm compiles the 
selected samples into a single data frame, returning it as a CSV file.

------------------------------------------------------------------------------
------------------------------------------------------------------------------
Created on Tue Oct 26 16:33:02 2021
@author: Ruben Remelgado
"""

#----------------------------------------------------------------------------#
# load modules and user arguments
#----------------------------------------------------------------------------#

from progress.bar import Bar
from argparse import ArgumentParser
from rasterio.mask import mask
from scipy.stats import skewnorm
from scipy import ndimage as nd
from shapely.geometry import shape
from os.path import join, dirname
import fiona as fn
import rasterio as rt
import pandas as pd
import numpy as np
import yaml

parser = ArgumentParser(description = 'Classification \
                        of canopy cover densities')
parser.add_argument('config', help='configuration file', type=str)
parser.add_argument('index', help='index of target tile', type=int)
parser.add_argument('start', help='initial sampling year', type=str)
parser.add_argument('end', help='last sampling year', type=str)

options = parser.parse_args()
config = options.config
index = options.index
start = options.start
end = options.end

# extract base directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))

# define output directories
ddir = join(wdir, config['data_dir'])
sdir = join(wdir, config["sample_dir"])

# access geocomputation boundaries
sp = fn.open(join(ddir, config['boundaries']['country_polygon']))

# define unique identifier for the tile
tid = ('{0:0' + len(str(len(sp))) + 'd}').format(index)

# extract bounding geometry from tile (used to subset input data)
geom = shape(sp[index]).bounds

#----------------------------------------------------------------------------#
# build sampling function
#----------------------------------------------------------------------------#

def main(v):
    
    #========================================================================#
    # sample changing pixels
    #========================================================================#
    
    # target number of samples
    class_samples = np.ceil(nr_samples*change[v]).astype('uint16')
    
    # find pixels with land cover v, forest losses, and no forest gains
    px1 = np.where((land_cover == unique_classes[v]) & \
                   (forest_change > 0) & (forest_gains == 0))
    
    # pick samples if nr. of indices is larger than nr. of samples
    if len(px1[0]) > class_samples:
        
        # if there is more than one sample
        if class_samples > 1:
            
            si = np.argsort(forest_change[px1]) # sort by change magnitude
            mx = len(si)-1 # maximum value of the skewed seq. to estimate
            ri = skewnorm.rvs(a=-5,loc=mx,size=class_samples) # right-skewed seq.
            ri = ri - min(ri) # Shift values so the minimum is equal to zero.
            ri = ri / max(ri) # Standadize all the vlues between 0 and 1. 
            ri = ri * mx # Multiply the standardized values by their maximum
            si = si[np.round(ri).astype('uint64')] # subset indices
            
        else:
            
            # if there is only one sample to choose, sample the maximum change
            si = np.flip(np.argsort(forest_change[px1]))[0].astype('uint64')
        
        # subset of samples
        px1 = (px1[0][si],px1[1][si])
    
    #========================================================================#
    # sample stable pixels
    #========================================================================#
    
    # target number of samples
    class_samples = np.ceil(nr_samples*nochange[v]).astype('uint16')
    
    # find pixels with land cover v, no forest losses, and no forest gains
    px2 = np.where((land_cover == unique_classes[v]) & \
                   (forest_change == 0) & (forest_gains == 0))
    
    # if there are more indices than desired samples, sample randomly
    if len(px2[0]) > class_samples:
        si = np.random.uniform(0,len(px2[0])-1,class_samples).astype('uint64')
        px2 = (px2[0][si],px2[1][si])
    
    #========================================================================#
    # translate selected indices into coordinates
    #========================================================================#
    
    if type(px1[0]).__name__ == 'int64':
        px1 = (np.array([px1[0]]), np.array([px1[1]]))
    
    if type(px2[0]).__name__ == 'int64':
        px2 = (np.array([px2[0]]), np.array([px2[1]]))
    
    # derive coordinates and add to final sample set
    x = list((xy[0] + pr/2) + px1[1] * pr) + list((xy[0] + pr/2) + px2[1] * pr)
    y = list((xy[1] - pr/2) - px1[0] * pr) + list((xy[1] + pr/2) - px2[0] * pr)
    
    # return output as a data frame
    return(pd.DataFrame({'x':x,'y':y}))

#----------------------------------------------------------------------------#
bar = Bar('access required datasets', max=5)
#----------------------------------------------------------------------------#

# mask limiting the pixels available for sampling
land_mask = rt.open(join(ddir, config['land_mask']))
pr = land_mask.res[0] # pixel resolution
land_mask, ot = mask(land_mask, geom, crop=True, pad=True, indexes=1)
xy = rt.transform.xy(ot, 0,0) # starting coordinates of window
bar.next()

# land cover, needed for stratified sampling
bname = start.join(config['variables']['predictors']['land_cover'].split('*'))
land_cover = mask(rt.open(join(ddir, bname)), geom, crop=True, pad=True, indexes=1)[0]
bar.next()

# GFC forest cover of initial year
bname = start.join(config['variables']['canopy_cover'].split('*'))
start_tcc = mask(rt.open(join(ddir, bname)), geom, crop=True, pad=True, indexes=1)[0]
bar.next()

# GFC forest cover of last year
bname = end.join(config['variables']['canopy_cover'].split('*'))
end_tcc = mask(rt.open(join(ddir, bname)), geom, crop=True, pad=True, indexes=1)[0]
bar.next()

# forest gains across the GFC time series
bname = start.join(config['variables']['forest_gains'].split('*'))
forest_gains = mask(rt.open(join(ddir, bname)), geom, crop=True, pad=True, indexes=1)[0]
bar.next()

bar.finish()

# estimate changes, so that values are positive
forest_change = start_tcc.astype('float32') - end_tcc.astype('float32')
del start_tcc
del end_tcc

#----------------------------------------------------------------------------#
# determine out many samples to collect
#----------------------------------------------------------------------------#

# determine the number of samples to select (min. 1 sample per per 50-km2)
region_size = np.sum((land_mask == 1) & (forest_gains == 0) & (land_cover > 0))
nr_samples = np.ceil(region_size/2500/2).astype('uint64')

# samples per land cover class
unique_classes, class_counts = np.unique(land_cover[(land_mask == 1) & \
                              (land_cover > 0)], return_counts=True)

# percentage of samples per land cover class (with forest change)
change = nd.sum(forest_change > 0, \
                labels=land_cover, index=unique_classes).astype('uint64')
change = change / np.sum(change)

# percentage of samples per land cover class (without forest change)
nochange = nd.sum(forest_change == 0, \
                  labels=land_cover, index=unique_classes).astype('uint64')
nochange = nochange / np.sum(nochange)

#----------------------------------------------------------------------------#
bar = Bar('extract samples', max=len(unique_classes))
#----------------------------------------------------------------------------#

odf = []

for v in range(0,len(unique_classes)):
    odf = odf + [main(v)]
    bar.nxt()

odf = pd.concat(odf)
bar.finish()

# write samples
odf.to_csv(join(sdir, f'{tid}_samples.csv'), index=False)
