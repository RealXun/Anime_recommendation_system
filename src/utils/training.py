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

# deal with sparse data libraries
from scipy.sparse import csr_matrix # Returns a copy of column i of the matrix, as a (m x 1) CSR matrix (column vector).

## scikit Preprocessing data libraries
from sklearn.preprocessing import MinMaxScaler # Transform features by scaling each feature to a given range.

# Python scikit for building and analyzing recommender systems that deal with explicit rating data
from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy

## scikit Cross validation iterators libraries
from sklearn.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

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



#############################################################
#############################################################
#                                                           #
#--------- Unsupervised User  based recommendation ---------#
#                                                           #
#############################################################
#############################################################

'''
The function matrix_creation_and_training converts a pivot table 
of user-item ratings into a sparse matrix using the csr_matrix function. 
It then fits a kNN model on this matrix using NearestNeighbors, 
and saves the model to a file using the pickle module. This process 
is an unsupervised learning technique for recommendation systems, 
where the goal is to identify similar items or users based on their ratings.
'''
def matrix_creation_and_training(df_pivot):
    # Convert pivot table of user-item ratings to a sparse matrix in CSR format
    matrix = csr_matrix(df_pivot.values)

    # Create k-Nearest Neighbors model with 2 neighbors, Euclidean distance metric, brute force algorithm, and p-norm=2
    model_knn = NearestNeighbors(n_neighbors=2, metric='euclidean', algorithm='brute', p=2)

    # Fit k-Nearest Neighbors model on the user-item rating matrix
    model_knn = model_knn.fit(matrix)

    # Save the trained k-Nearest Neighbors model to a file using the pickle module
    
    with open(saved_models_folder + "/" + "nearest_user_base_new_model.pkl", "wb") as f:
        pickle.dump(model_knn, f)

    # Return the trained k-Nearest Neighbors model
    return model_knn




##############################################################
##############################################################
#                                                            #
#----------- Supervised User based recommendation -----------#
#                                                            #
##############################################################
##############################################################

'''
In this code, the data is split into training and testing sets using 
the train_test_split() function from surprise library. Then, an instance 
of the SVD algorithm is created with the best parameters obtained 
from the grid search, and it is trained on the training set using the fit() method.
'''
def train_test_svd():
 
        # Load model with best parameters
        gs = joblib.load(saved_models_folder + "/" + "SVD_model_best_params.pkl")
        data = joblib.load(processed_data + "/" + "data_reader_sample.pkl")    

        # Split data into training and testing sets
        trainset, testset = train_test_split(data, test_size=0.2)       
        # Train SVD algorithm on training set with best parameters
        best_params = SVD(n_factors=gs.best_params['rmse']['n_factors'], 
                n_epochs=gs.best_params['rmse']['n_epochs'], 
                lr_all=gs.best_params['rmse']['lr_all'], 
                reg_all=gs.best_params['rmse']['reg_all'])
        best_params.fit(trainset)       
        # Make predictions on testing set
        predictions = best_params.test(testset) 
        # Calculate RMSE and MAE
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions) 
        print("RMSE:", rmse)
        print("MAE:", mae)      
        # # Serialización del modelo
        import pickle
        joblib.dump(best_params,saved_models_folder + "/" + "SVD_new_model.pkl")


