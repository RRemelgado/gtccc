# -*- coding: utf-8 -*-
""" model quality assessment
------------------------------------------------------------------------------
This algorithm uses spatial cross-validation to evaluate the performance 
of our model, which was designed to predict tree-canopy cover. The algorithm 
iterates through a 

------------------------------------------------------------------------------
------------------------------------------------------------------------------
Created on Wed Jul 20 07:50:44 2022
@author: Ruben Remelgado
"""

#----------------------------------------------------------------------------#
# load modules and arguments
#----------------------------------------------------------------------------#

from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
from os.path import join, dirname
import pandas as pd
import numpy as np
import yaml

parser = ArgumentParser(description = 'test model quality')
parser.add_argument('config', help='configuration file', type=str)
parser.add_argument('forest', 'number of trees', type=int)
parser.add_argument('index', 'region index', type=int)

options = parser.parse_args()
config = options.config
forest = options.forest
index = options.index

# extract base directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))
vdir = config['quality_dir']

#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#

x = pd.read_csv(join(wdir, config["model_dir"], "x_data.csv"))
y = pd.read_csv(join(wdir, config["model_dir"], "y_data.csv"))
predictors = x.keys()

# target region
subregion = np.unique(y['district'].values)[index]

#----------------------------------------------------------------------------#
# collect sampling and validation indices
#----------------------------------------------------------------------------#

# indices of samples for the target subregion (used to validate a model)
val = np.where(y['district'] == subregion)

# region where samples were collected
country = np.unique(y['country'].iloc[val])[0]

# indices of samples in region (used to train a model)
trn = np.where((y['district'] != subregion) & (y['country'] != subregion))

#----------------------------------------------------------------------------#
# build predictive model
#----------------------------------------------------------------------------#

model = RandomForestRegressor(n_estimators=forest).fit(
    x.iloc[trn],
    y['target'].values[trn].copy().flatten()) # fit model

#----------------------------------------------------------------------------#
# validate  model
#----------------------------------------------------------------------------#

# collecte data on predicted samples
rid = [country]*len(val[0]) # subregion name/ID
mv = model.predict(x.iloc[val].copy())
prd_mv = list(mv) # predicition

#----------------------------------------------------------------------------#
# find confidence interval (95%)
#----------------------------------------------------------------------------#

prediction = np.zeros(x.shape[0], dtype='float32')
for e in range(0,forest):
    prediction = prediction + (mv-model.estimators_[e].predict(x))**2

ci = 1.96 * np.sqrt(prediction/ len(model.estimators_)) / np.sqrt(forest)

del model

#----------------------------------------------------------------------------#
# recond prediction and confidence interval, and 
#----------------------------------------------------------------------------#

y['prediction'] = prd_mv
y['confidence'] = ci
tid = '{0:04d}'.format(index)
y.to_csv(join(vdir, f'{tid}_quality_assessment.csv'), index=False)
