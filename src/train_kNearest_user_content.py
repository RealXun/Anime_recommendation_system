# Standard library imports
import os # allows access to OS-dependent functionalities
import pandas as pd
from pathlib import Path

# Third party libraries

import joblib # set of tools to provide lightweight pipelining in Python

# Unsupervised learner for implementing neighbor searches.
from sklearn.neighbors import NearestNeighbors


pd.options.mode.chained_assignment = None  # default='warn'

#Preparing folder variables
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)

data_folder = (PROJECT_ROOT + "/" + "data")
saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")


############################################################
############################################################
#                                                          #
#----- Unsupervised User content based recommendation -----#
#                                                          #
############################################################
############################################################

'''
The function model_NearestNeighbors builds and trains a 
k-Nearest Neighbors model on a given dataset, using specified 
parameters. It then saves the indices of the nearest neighbors 
to a file and returns them.
'''

def model_NearestNeighbors(df):
    # Build and "train" the model using NearestNeighbors algorithm
    # algorithm: algorithm used to compute the nearest neighbors (‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’)
    # leaf_size: leaf size passed to BallTree or KDTree
    # metric: distance metric used for the tree. Can be 'minkowski', 'euclidean', etc.
    # n_neighbors: number of neighbors to use for kneighbors queries
    # p: power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance
    neigh = NearestNeighbors(algorithm= 'auto', leaf_size= 30, metric= 'minkowski', n_neighbors= 100, p= 1, radius= 0.0).fit(df)

    # Get the distances and indices of the nearest neighbors
    # distances: array representing the lengths to points, only present if return_distance=True
    # indices: indices of the nearest points in the population matrix
    distances, indices = neigh.kneighbors(df)

    # Save the model to a file using joblib.dump
    joblib.dump(indices, saved_models_folder + "/" + "kNearest_user_content_new_model.pkl")

    # Return the indices of the nearest neighbors
    return indices