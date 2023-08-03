# -*- coding: utf-8 -*-
""" build_model
------------------------------------------------------------------------------
Builds Random Forest (RF) model to predict tree cover percentages using 
all available samples. The mdoel uses bootstrapingm, and the number of 
trees built by the model is specified in the input configuration file.
------------------------------------------------------------------------------
------------------------------------------------------------------------------
Created on Tue Nov  2 14:52:31 2021
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
import pickle
import yaml

parser = ArgumentParser(description = 'test modelling of forest cover')
parser.add_argument('config', 'path to configuration yaml')
parser.add_argument('forest', 'number of trees')

options = parser.parse_args()
config = options.config
forest = options.forest

# extract base directory
wdir = dirname(config)

# load parameters
config = yaml.safe_load(open(config, "r"))
mdir = join(wdir, config["model_dir"])

#----------------------------------------------------------------------------#
bar = Bar('build final model (uses all samples)', max=4)
#----------------------------------------------------------------------------#

x = pd.read_csv(join(wdir, config["model_dir"], "x_data.csv"))
y = pd.read_csv(join(wdir, config["model_dir"], "y_data.csv"))['target'].values
variables = x.columns

bar.next()

model = RandomForestRegressor(n_estimators=forest).fit(x,y)

bar.next()

sdf = pd.DataFrame({'variables':variables, 'importance':model.feature_importances_})
sdf.to_csv(join(mdir, 'variable_importance.csv'), index=False)

bar.next()

pickle.dump(model, open(join(mdir, 'model.pkl'), 'wb'))

bar.next()
bar.finish()
