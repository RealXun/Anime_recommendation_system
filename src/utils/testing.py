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

from surprise import SVD
from surprise.model_selection import GridSearchCV

## scikit Cross validation iterators libraries
from sklearn.model_selection import GridSearchCV
from surprise.model_selection import cross_validate

# Unsupervised learner for implementing neighbor searches.
from sklearn.neighbors import NearestNeighbors

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
test_models = (saved_models_folder + "/" + "test_models")



############################################################
############################################################
#                                                          #
#----- Unsupervised User content based recommendation -----#
#                                                          #
############################################################
############################################################


def param_NearestNeighbors(df):
    '''
    The function param_NearestNeighbors uses GridSearchCV from scikit-learn 
    to perform a grid search over a range of hyperparameters for the 
    NearestNeighbors model. It takes a dataframe df as input and returns 
    the best hyperparameters found during the grid search. The hyperparameters 
    being searched over include n_neighbors, radius, algorithm, leaf_size, 
    metric, and p. The scoring metric being used is "accuracy" and the refit 
    parameter is set to "precision_score". cv=2 sets the number of cross-validation 
    folds to 2, and n_jobs=-1 sets the number of CPU cores used to parallelize 
    the search to be the maximum available.
    '''
    # Define dictionary of hyperparameters to test using GridSearchCV
    parametros = {
        'n_neighbors': range(1, 10),
        'radius': np.linspace(0, 1, 11),
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
        'p': range(1, 2)
    }

    # Create GridSearchCV object with NearestNeighbors algorithm
    # and hyperparameters defined in the parametros dictionary
    model_knn = GridSearchCV(NearestNeighbors(), parametros, scoring='accuracy', refit='precision_score', cv=2, n_jobs=-1).fit(df)

    # Return the best hyperparameters found by the grid search
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

