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
from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold

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

def mape(predictions):
    """
    Compute the Mean Absolute Percentage Error (MAPE) for a set of predictions.
    
    Args:
        predictions: list of Prediction objects returned by the test method of an algorithm
        
    Returns:
        The MAPE score
    """
    actual_ratings = np.array([pred.r_ui for pred in predictions])
    predicted_ratings = np.array([pred.est for pred in predictions])
    return np.mean(np.abs(actual_ratings - predicted_ratings) / actual_ratings) * 100

def r2(predictions):
    """
    Compute the R-squared (R2) score for a set of predictions.
    
    Args:
        predictions: list of Prediction objects returned by the test method of an algorithm
        
    Returns:
        The R2 score
    """
    actual_ratings = np.array([pred.r_ui for pred in predictions])
    predicted_ratings = np.array([pred.est for pred in predictions])
    mean_rating = np.mean(actual_ratings)
    ss_tot = np.sum((actual_ratings - mean_rating) ** 2)
    ss_res = np.sum((actual_ratings - predicted_ratings) ** 2)
    return 1 - (ss_res / ss_tot)


def train_test_svd():
    '''
    In this code, the data is split into training and testing sets using 
    the train_test_split() function from surprise library. Then, an instance 
    of the SVD algorithm is created with the best parameters obtained 
    from the grid search, and it is trained on the training set using the fit() method.
    '''
    # Loads the best hyperparameters for the SVD algorithm that were obtained from grid search
    gs = joblib.load(saved_models_folder + "/" + "SVD_new_model_best_params.pkl")

    # Loads the dataset from a pickle file using joblib
    data = joblib.load(processed_data + "/" + "data_reader_sample.pkl")    

    # Splits the data into training and testing sets with a 80:20 ratio
    trainset, testset = train_test_split(data, test_size=0.2)       

    # Creates an instance of the SVD algorithm with the best hyperparameters obtained from grid search
    best_params = SVD(n_factors=gs.best_params['rmse']['n_factors'], 
                    n_epochs=gs.best_params['rmse']['n_epochs'], 
                    lr_all=gs.best_params['rmse']['lr_all'], 
                    reg_all=gs.best_params['rmse']['reg_all'])

    # Trains the SVD algorithm on the training set using the fit() method
    best_params.fit(trainset)       

    # Generates predictions for the test set using the trained model
    predictions = best_params.test(testset) 

    # Calculates the RMSE and MAE for the predictions
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions) 
    mse = accuracy.mse(predictions)
    #mape_score = mape(predictions)
    #r2_score = r2(predictions)
    print("RMSE:", rmse)
    print("MSE:", mse)  
    print("MAE:", mae)  
    #print("mape_score:", mape_score)  
    #print("r2_score:", r2_score)    

    # Saves the trained model as a pickle file using joblib
    joblib.dump(best_params,saved_models_folder + "/" + "SVD_new_model.pkl")




def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls




def svd_precision_recall():
    '''
    In this code, the data is split into training and testing sets using 
    the train_test_split() function from surprise library. Then, an instance 
    of the SVD algorithm is created with the best parameters obtained 
    from the grid search, and it is trained on the training set using the fit() method.
    '''
    # Loads the best hyperparameters for the SVD algorithm that were obtained from grid search
    gs = joblib.load(saved_models_folder + "/" + "SVD_best_params_test_model.pkl")

    # Loads the dataset from a pickle file using joblib
    data = joblib.load(processed_data + "/" + "data_reader_sample.pkl")    

    # Splits the data into training and testing sets with a 80:20 ratio
    trainset, testset = train_test_split(data, test_size=0.2)       

    # Creates an instance of the SVD algorithm with the best hyperparameters obtained from grid search
    algo = SVD(n_factors=gs.best_params['rmse']['n_factors'], 
                    n_epochs=gs.best_params['rmse']['n_epochs'], 
                    lr_all=gs.best_params['rmse']['lr_all'], 
                    reg_all=gs.best_params['rmse']['reg_all'])  

    kf = KFold(n_splits=5)  # initialize a KFold object with 5 splits

    count = 1  # initialize a count variable
    precision_list = []  # initialize an empty list to store precision scores
    recall_list = []  # initialize an empty list to store recall scores
    f1_score_list = []  # initialize an empty list to store F1 scores

    # loop through each split of the data using the KFold object
    for trainset, testset in kf.split(data):

        algo.fit(trainset)  # fit the recommendation algorithm on the training set

        predictions = algo.test(testset)  # make predictions on the test set using the algorithm

        # calculate the precision and recall at k
        precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

        # calculate the precision, recall, and F1 score for the split
        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # append the scores to their respective lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)            

        # print the scores for the current split
        print('K =', count, '--- Precision:', precision, '--- Recall:', recall, '--- F1 score:',f1_score)
        count +=1

    # return a dictionary containing the lists of scores
    return {'precision': precision_list, 'recall': recall_list, 'f1_score': f1_score_list}


