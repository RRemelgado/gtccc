# -*- coding: utf-8 -*-
"""select_nr_trees
This algorithm finds the optimal number of trees to calibrate a 
Random Forest (RF) model intended to predicte tree-canopy cover. It 
iterates through different numbers of trees, between the minimum and 
maximum specified in the configuration file, which also specified the 
step size. At each iteration, a RF model is fit using a randomly selected 
number of samples, and the Out-of-Bag (OOB) error is reported as a metric 
of quality. The output, reporting on the OOB error per number of trees, is 
reported as a CSV, following the path specified in the configuration file. 
------------------------------------------------------------------------------
------------------------------------------------------------------------------
Created on Tue Nov  2 15:23:47 2021
@author: Ruben Remelgado
"""
#----------------------------------------------------------------------------#
# load modules and configuration data
#----------------------------------------------------------------------------#

from progress.bar import Bar
from os.path import dirname, join
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import random
import yaml

parser = ArgumentParser(description = 'Find optimal number of trees')
parser.add_argument('config', 'path to configuration yaml')
options = parser.parse_args()
config = options.config

# extract parent directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))
mdir = join(wdir, config["model_dir"])

# read samples
x_data = pd.read_csv(join(mdir, "x_data.csv"))
x_data = pd.read_csv(join(mdir, "y_data.csv"))

# configure search for optimal number of trees
# build array to test number of trees
step = config["forest_size"]["step_size"]
start = config["forest_size"]["min"]
end = config["forest_size"]["max"]
nr_samples = config["forest_size"]["samples"]
nr_runs = config["forest_size"]["runs"]
nr_trees = np.arange(start,end+step,step)

# variables used to train a model
predictors = list(config['variables']['predictors'].keys()) + ['gap']

#----------------------------------------------------------------------------#
bar = Bar('test candidate number of trees', max=len(nr_trees)*nr_runs)
#----------------------------------------------------------------------------#

# Out-of-Bag (OOB) container (rows are used for trees and columns for runs)
oob = np.zeros((len(nr_trees), nr_runs), dtype='float32')

# iterated through r runs and n tree sizes
for r in range(0,nr_runs):
    
    # generate random indices of samples
    ind = np.random.uniform(0,x_data.shape[0],x_data).astype('uint64')
    
    x = x_data[predictors].iloc[ind] # subset of predictors
    y = x_data['target'].values[ind].flatten() # subset of target
    seed = random.seed(1234)
    
    # evaluate each target number of trees
    for n in range(0,len(nr_trees)):
        
        # build model
        model = RandomForestRegressor(
            oob_score=True, 
            bootstrap=True, 
            n_estimators=nr_trees[n], 
            random_state=seed
            ).fit(x,y)
        
        # record OOB
        oob[n,r] = model.oob_score_
        bar.next()
  
bar.finish()

#----------------------------------------------------------------------------#
# write test results
#----------------------------------------------------------------------------#

odf = pd.DataFrame({'nr_trees':nr_trees, \
                    'oob_mean':oob.mean(axis=1), \
                        'oob_sd':oob.std(axis=1)})
odf.to_csv(join(mdir, 'nr_trees-summary.csv'), index=False)

oob = pd.DataFrame(data=oob, columns=['run' + str(r) for r in range(0,nr_runs)])
oob.to_csv(join(mdir, 'nr_trees-all_runs.csv'), index=False)
