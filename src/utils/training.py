# Standard library imports
import os # allows access to OS-dependent functionalities
import re #  regular expression matching operations similar to those found in Perl
import sys # to manipulate different parts of the Python runtime environment
import warnings # is used to display the message Warning
import pickle # serializing and deserializing a Python object structure.

# Third party libraries
from fastparquet import write # parquet format, aiming integrate into python-based big data work-flows
from fuzzywuzzy import fuzz # used for string matching

from collections import defaultdict
import numpy as np # functions for working in domain of linear algebra, fourier transform, matrices and arrays
import pandas as pd # data analysis and manipulation tool
import joblib # set of tools to provide lightweight pipelining in Python

# deal with sparse data libraries
from scipy.sparse import csr_matrix # Returns a copy of column i of the matrix, as a (m x 1) CSR matrix (column vector).

## scikit Preprocessing data libraries
from sklearn.preprocessing import MinMaxScaler # Transform features by scaling each feature to a given range.

# Python scikit for building and analyzing recommender systems that deal with explicit rating data
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate,KFold

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



def model_NearestNeighbors(df):
    '''
    The function model_NearestNeighbors builds and trains a 
    k-Nearest Neighbors model on a given dataset, using specified 
    parameters. It then saves the indices of the nearest neighbors 
    to a file and returns them.
    '''

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


def matrix_creation_and_training(df_pivot):
    '''
    The function matrix_creation_and_training converts a pivot table 
    of user-item ratings into a sparse matrix using the csr_matrix function. 
    It then fits a kNN model on this matrix using NearestNeighbors, 
    and saves the model to a file using the pickle module. This process 
    is an unsupervised learning technique for recommendation systems, 
    where the goal is to identify similar items or users based on their ratings.
    '''
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
def train_test_svd(size=None):
    '''
    In this code, the data is split into training and testing sets using 
    the train_test_split() function from surprise library. Then, an instance 
    of the SVD algorithm is created with the best parameters obtained 
    from the grid search, and it is trained on the training set using the fit() method.
    '''
    if size:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_1million.pkl")

        # Loads the best hyperparameters for the SVD algorithm that were obtained from grid search
        gs = joblib.load(test_models + "/" + "SVD_best_params_test_model_1million.pkl")        
            
    else:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_whole_data.pkl")

        # Loads the best hyperparameters for the SVD algorithm that were obtained from grid search
        gs = joblib.load(test_models + "/" + "SVD_best_params_test_model_whole_data.pkl")       


    # Splits the data into training and testing sets with a 80:20 ratio
    trainset, testset = train_test_split(data, test_size=0.2)       

    # Creates an instance of the SVD algorithm with the best hyperparameters obtained from grid search
    best_params = SVD(n_factors=gs['rmse']['n_factors'], 
                    n_epochs=gs['rmse']['n_epochs'], 
                    lr_all=gs['rmse']['lr_all'], 
                    reg_all=gs['rmse']['reg_all'])

    # Evaluate the model using cross-validation
    cv_results = cross_validate(best_params, data, measures=['RMSE', 'MAE', 'MSE','FCP'], cv=5, verbose=True)

    # Print the average RMSE and MAE across all folds
    print('Average RMSE Training:', round(cv_results['test_rmse'].mean(), 2))
    print('Average MAE Training:', round(cv_results['test_mae'].mean(), 2))
    print('Average MSE Training:', round(cv_results['test_mse'].mean(), 2))
    print('Average FCP Training:', cv_results['test_fcp'].mean())

    # Trains the SVD algorithm on the training set using the fit() method
    best_params.fit(trainset)       
    
    # Generates predictions for the test set using the trained model
    predictions = best_params.test(testset) 

    # Calculates the RMSE, MSE and MAE for the predictions
    rmse = round(accuracy.rmse(predictions, verbose=False), 3)
    mse = round(accuracy.mse(predictions, verbose=False), 3)
    mae = round(accuracy.mae(predictions, verbose=False), 3)
    fcp = accuracy.fcp(predictions, verbose=False)

    # Print the results of the RMSE, MSE and MAE for the predictions
    print(f"RMSE Test: {rmse:.3f}")
    print(f"MSE Test: {mse:.3f}")
    print(f"MAE Test: {mae:.3f}")
    print(f"FCP Test: {fcp}")

    # Saves the trained model as a pickle file using joblib
    joblib.dump(predictions,saved_models_folder + "/" + "SVD_model_predictions.pkl")


    # Saves the trained model as a pickle file using joblib
    joblib.dump(best_params,saved_models_folder + "/" + "SVD_new_model.pkl")

    # Compresses the pickle file using zip and saves it
    dir, base_filename = os.path.split(saved_models_folder + "/" + "SVD_new_model.pkl")
    os.chdir(dir)
    import zipfile as ZipFile
    import zipfile
    zip = zipfile.ZipFile('SVD_new_model.zip',"w", zipfile.ZIP_DEFLATED)
    zip.write(base_filename)
    zip.close()