# Standard library imports
import os # allows access to OS-dependent functionalities
import pandas as pd
from pathlib import Path
import numpy as np

# Third party libraries

import joblib # set of tools to provide lightweight pipelining in Python

# Unsupervised learner for implementing neighbor searches.
from sklearn.neighbors import NearestNeighbors

## scikit Preprocessing data libraries
from sklearn.preprocessing import MinMaxScaler # Transform features by scaling each feature to a given range.

# Utils libraries
from utils import cleaning
from utils import recommend
from utils import testing
from utils import training

pd.options.mode.chained_assignment = None  # default='warn'

#Preparing folder variables
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)

src_folder = (PROJECT_ROOT + "/" + "src")
data_folder = (src_folder + "/" + "data")
saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")
test_models = (saved_models_folder + "/" + "test_models")


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
print('\nOpening CSV file called "anime.csv"...')
# CSV file called "anime.csv" from a directory called raw_data and returns the contents as a Pandas DataFrame
anime = pd.read_csv(raw_data + "/" + "anime.csv") 
print('Done.')

print('\nOpening CSV file called "rating.csv.zip"...')
# CSV file called "rating.csv.zip" from a directory called raw_data and returns the contents as a Pandas DataFrame
rating = pd.read_csv(raw_data + "/" + "rating.csv.zip") 
print('Done.')

# Calling the cleaning functions
print('\nCalling the cleaning functions...')
anime_cleaned = cleaning.clean_anime_df(anime)
print('Done.')

# This function prepares the content-based features for a supervised learning model
print('\nPreparing the content-based features for a supervised learning model...')
anime_features = cleaning.prepare_supervised_content_based(anime_cleaned) 
print('Done.')

# Initialize MinMaxScaler object
print('\nInitializing MinMaxScaler object...')
min_max = MinMaxScaler()
print('Done.')


# Scale the anime features using MinMaxScaler
print('\nScaling the anime features using MinMaxScaler...')
min_max_features = min_max.fit_transform(anime_features)
print('Done.')


# Round the scaled features to 2 decimal places using numpy
print("\nRounding the scaled features to 2 decimal places...")
np.round(min_max_features,2)
print('Done.')


# Build and "train" the model using NearestNeighbors algorithm
# algorithm: algorithm used to compute the nearest neighbors (???auto???, ???ball_tree???, ???kd_tree???, ???brute???)
# leaf_size: leaf size passed to BallTree or KDTree
# metric: distance metric used for the tree. Can be 'minkowski', 'euclidean', etc.
# n_neighbors: number of neighbors to use for kneighbors queries
# p: power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance
print("Building and training the model using NearestNeighbors algorithm...")
neigh = NearestNeighbors(algorithm= 'auto', leaf_size= 30, metric= 'minkowski', n_neighbors= 100, p= 1, radius= 0.0).fit(min_max_features)
print('Done.')

# Get the distances and indices of the nearest neighbors
# distances: array representing the lengths to points, only present if return_distance=True
# indices: indices of the nearest points in the population matrix
print("Getting the distances and indices of the nearest neighbors...")
distances, indices = neigh.kneighbors(min_max_features)
print('Done.')

# Save the model to a file using joblib.dump
print("\nSaving trained model...")
joblib.dump(indices, saved_models_folder + "/" + "kNearest_user_content_new_model.pkl")
print("Model saved successfully!") 