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



############################################################
############################################################
#                                                          #
#----- Unsupervised User content based recommendation -----#
#                                                          #
############################################################
############################################################

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

def param_NearestNeighbors(df):
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

'''
This function evaluates the SVD algorithm using GridSearchCV to find 
the best combination of hyperparameters based on the provided parameter grid. 
It prints the best RMSE and MAE scores and their corresponding parameters, 
and returns a results dataframe.
'''
def evaluate_svd_model(data_sample):
    # Define parameter grid for grid search
    param_grid = {'n_factors': [50, 100, 150], 
                'n_epochs': [20, 30, 40], 
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.05, 0.1]}
    
    # Create GridSearchCV object with SVD algorithm
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

    # Fit GridSearchCV object to data
    gs.fit(data)

    # Print best RMSE and MAE scores, as well as corresponding parameters
    print("Best RMSE score:", gs.best_score['rmse'])
    print("Best MAE score:", gs.best_score['mae'])
    print("Best parameters for RMSE:", gs.best_params['rmse'])
    print("Best parameters for MAE:", gs.best_params['mae'])

    return results_df



'''
defines a parameter grid for hyperparameter tuning in a collaborative filtering algorithm.
Then create a GridSearchCV object with the SVD algorithm and a parameter grid consisting 
of a range of hyperparameters. The GridSearchCV function then performs a grid search on 
yhe parameter grid to find the best combination of hyperparameters that minimizes the 
RMSE and MAE scores. The best RMSE and MAE scores and the corresponding parameters 
are printed out.
'''

def find_best_svd():
    from surprise import SVD
    from surprise.model_selection import GridSearchCV
    data = joblib.load(processed_data + "/" + "data_reader_sample.pkl")

    # Define parameter grid for grid search
    param_grid = {'n_factors': [50, 100, 150], 
                'n_epochs': [20, 30, 40], 
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.05, 0.1]}

    # Create GridSearchCV object with SVD algorithm
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

    # Fit GridSearchCV object to data
    gs.fit(data)

    # Print best RMSE and MAE scores, as well as corresponding parameters
    print("Best RMSE score:", gs.best_score['rmse'])
    print("Best MAE score:", gs.best_score['mae'])
    print("Best parameters for RMSE:", gs.best_params['rmse'])
    print("Best parameters for MAE:", gs.best_params['mae'])

    # Save model with best parameters
    joblib.dump(gs,saved_models_folder + "/" + "SVD_model_best_params.pkl")

    # Save best parameters
    joblib.dump(gs.best_params,saved_models_folder + "/" + "best_params_for_SVD.pkl", compress = 1)

    return gs