# Standard library imports
import os # allows access to OS-dependent functionalities
import re #  regular expression matching operations similar to those found in Perl
import sys # to manipulate different parts of the Python runtime environment
import warnings # is used to display the message Warning
import pickle # serializing and deserializing a Python object structure.

# Third party libraries
from fastparquet import write # parquet format, aiming integrate into python-based big data work-flows
from fuzzywuzzy import fuzz # used for string matching

import numpy as np # functions for working in domain of linear algebra, fourier transform, matrices and arrays
import pandas as pd # data analysis and manipulation tool
import joblib # set of tools to provide lightweight pipelining in Python

## scikit Preprocessing data libraries
from sklearn.preprocessing import MinMaxScaler # Transform features by scaling each feature to a given range.

# Python scikit for building and analyzing recommender systems that deal with explicit rating data
from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy

## scikit Cross validation iterators libraries
from sklearn.model_selection import GridSearchCV
from surprise.model_selection import cross_validate

# Unsupervised learner for implementing neighbor searches.
from sklearn.neighbors import NearestNeighbors

# Utils libraries
from utils import utils
from utils import cleaning
from utils import recommend
from utils import testing
from utils import training

pd.options.mode.chained_assignment = None  # default='warn'

#Preparing folder variables
#os.chdir(os.path.dirname(sys.path[0])) # This command makes the notebook the main path and can work in cascade.
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
data_folder = (PROJECT_ROOT + "/" + "data")

saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")
content_based_supervised_data = (data_folder + "/" + "processed" + "/" + "content_based_supervised")



############################################################
############################################################
#                                                          #
#----- Unsupervised User content based recommendation -----#
#                                                          #
############################################################
############################################################

def param_NearestNeighbors(df):

    parametros = { 'n_neighbors' : range(1,10),
               'radius' : np.linspace(0,1,11),
               'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
			   'leaf_size' : [30],
			   'metric' : ['minkowski','euclidean','manhattan'],
			   'p' : range(1,2)
			   }
    model_knn = GridSearchCV(NearestNeighbors(), parametros, scoring="accuracy",refit='precision_score', cv=2, n_jobs = -1).fit(df)

    return model_knn.best_params_


#############################################################
#############################################################
#                                                           #
#--------- Unsupervised User  based recommendation ---------#
#                                                           #
#############################################################
#############################################################





##############################################################
##############################################################
#                                                            #
#----------- Supervised User based recommendation -----------#
#                                                            #
##############################################################
##############################################################

def evaluate_svd_model(data_sample):
    param_grid = {'n_factors':[50,100,150],'n_epochs':[20,30],  'lr_all':[0.005,0.01],'reg_all':[0.02,0.1]}
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

    gs.fit(data_sample)
    params = gs.best_params['rmse']

    # best RMSE score
    print(gs.best_score["rmse"])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params["rmse"])

    # We can now use the algorithm that yields the best rmse:
    algo = gs.best_estimator["rmse"]
    algo.fit(data_sample.build_full_trainset())

    # # Serialización del modelo
    joblib.dump(algo,saved_models_folder + "/" + "SVD_samople_fit.pkl")

    results_df = pd.DataFrame.from_dict(gs.cv_results)
    
    return results_df