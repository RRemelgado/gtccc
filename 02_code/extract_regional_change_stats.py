# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:38:40 2023

@author: rr70wedu
"""

#----------------------------------------------------------------------------#
# load modules and user arguments
#----------------------------------------------------------------------------#

from argparse import ArgumentParser
from progress.bar import Bar
from rasterio.mask import mask
from os.path import join
import rasterio as rt
import pandas as pd
import numpy as np
import fiona as fn

parser = ArgumentParser(description = 'regional variable')
parser.add_argument("variable", help = "target variable")

options = parser.parse_args()
variable = options.variable

wdir = '/data/idiv_meyer/01_projects/Ruben/GlobES/tmp/reconstruct_forest/'

sp = fn.open(join(wdir, '00_data', 'FRA_ecozones.shp'))

# ranges = [[2000,2018],[1992,2018],[1992,2000],[2000,2010],[2010,2018]]
ranges = [[1992,2018]]

pixel_area = rt.open(join(wdir, '00_data', 'pixelArea-hectares_20190000_300m.tif'))

#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#

for y in range(0, len(ranges)):
    
    period = str(ranges[y][0]) + '_' + str(ranges[y][1])
    
    #========================================================================#
    #
    #========================================================================#
    
    start_ds = rt.open(join(wdir, '01_analysis', 'map', 
                            f'iGFC-{variable}_{ranges[y][0]}0000_300m.tif'))
    end_ds = rt.open(join(wdir, '01_analysis', 'map', 
                          f'iGFC-{variable}_{ranges[y][1]}0000_300m.tif'))
    change_ds = rt.open(join(wdir, '01_analysis', 'change', 
                             f'iGFC-changeClass_{ranges[y][0]}0000-{ranges[y][1]}0000_300m.tif'))
    
    #========================================================================#
    bar = Bar(f'processing time period {period}', max=len(sp))
    #========================================================================#
    
    odf = []
    
    for i in range(0,len(sp)):
        
        #====================================================================#
        #
        #====================================================================#
        
        bb = [sp[i]['geometry']]
        
        biome = sp[i]['properties']['biome']
        region = sp[i]['properties']['region']
        
        #====================================================================#
        #
        #====================================================================#
        
        area = mask(pixel_area, bb, crop=True, pad=True, 
                    all_touched=True, nodata=0, indexes=1)[0]
        
        dims = area.shape
        
        change = mask(change_ds, bb, crop=True, pad=True, 
                      all_touched=True, indexes=1)[0][0:dims[0],0:dims[1]]
        
        #====================================================================#
        #
        #====================================================================#
        
        sa = np.round(mask(start_ds, bb, crop=True, pad=True, all_touched=True, 
                  nodata=0, indexes=1)[0].astype('float32')[0:dims[0],0:dims[1]])
        
        ea = np.round(mask(end_ds, bb, crop=True, pad=True, all_touched=True, 
                  nodata=0, indexes=1)[0].astype('float32')[0:dims[0],0:dims[1]])
        
        sa[sa > 100] = 0
        ea[ea > 100] = 0
        
        #====================================================================#
        #
        #====================================================================#
        
        for v, x in enumerate(['not forest', 'stable forest',
                               'loss','gain','disturbed'],start=1):
            
            px = np.where(change == v)
            
            if len(px[0]) > 0:
                class_area = np.sum(area[px]*((sa[px] > 0) | (ea[px] > 0)))
                px_loss_area = np.sum(area[px]*((sa[px] > 0) & (ea[px] == 0)))
                px_gain_area = np.sum(area[px]*((sa[px] == 0) & (ea[px] > 0)))
                start_cover_area = np.sum(np.mean(sa[px])*0.01*area[px])
                cover_area_change = np.sum(np.mean(ea[px])*0.01*area[px])-start_cover_area
            else:
                class_area = 0
                px_loss_area = 0
                px_gain_area = 0
                start_cover_area = 0
                cover_area_change = 0
            
            odf.append(pd.DataFrame(
                {
                    'fid':[i], 
                    'group_area':[class_area], 
                    'start_cover_area':[start_cover_area],
                    'cover_area_change':[cover_area_change], 
                    'extent_area_loss':[px_loss_area], 
                    'extent_area_gain':[px_gain_area], 
                    'region':[region], 
                    'biome':[biome], 
                    'group':[x],
                    'period':[period], 
                    'variable':[variable]
                    }
                ))
        
        bar.next()
    
    odf = pd.concat(odf)
    oname = join(wdir, '01_analysis', 'stats','regional', 
                 f'GFCC_change_{variable}_{period}.csv')
    odf.to_csv(oname, index=False)
    
    del odf
    bar.finish()
    print(period)
