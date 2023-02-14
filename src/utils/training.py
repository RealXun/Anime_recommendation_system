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
#data_folder = (PROJECT_ROOT + "\data")
#
#saved_models_folder = (data_folder + "\saved_models")
#raw_data = (data_folder + "\_raw")
#processed_data = (data_folder + "\processed")
#content_based_supervised_data = (data_folder + "\processed\content_based_supervised")

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
    # Build and "train" the model
    neigh = NearestNeighbors(algorithm= 'auto', leaf_size= 30, metric= 'minkowski', n_neighbors= 100, p= 1, radius= 0.0).fit(df)
    distances, indices = neigh.kneighbors(df)

    joblib.dump(indices,saved_models_folder + "/" + "model_based_content.pkl")

    return indices


#############################################################
#############################################################
#                                                           #
#--------- Unsupervised User  based recommendation ---------#
#                                                           #
#############################################################
#############################################################

def matrix_creation_and_training(df_pivot):

    matrix = csr_matrix(df_pivot.values)

    model_knn = NearestNeighbors(n_neighbors=2,metric = 'euclidean', algorithm = 'brute',p=2)
    model_knn = model_knn.fit(matrix)

    # Saving the pivot table to pickle
    fichero = open(saved_models_folder + "/" +"model_matrix_user_based_unsupervised.pkl","wb")
    pickle.dump(model_knn,fichero)
    fichero.close()

    return model_knn



##############################################################
##############################################################
#                                                            #
#----------- Supervised User based recommendation -----------#
#                                                            #
##############################################################
##############################################################

def supervised_prepare_training(ratingdf):
    # using groupby and some fancy logic
    reader = Reader(rating_scale=(1,10))
    data = Dataset.load_from_df(ratingdf[['user_id', 'anime_id', 'rating']], reader)
    
    size = 100000
    rating_sample = ratingdf.groupby("rating", group_keys=False).apply(lambda x: x.sample(int(np.rint(size*len(x)/len(ratingdf))))).sample(frac=1).reset_index(drop=True)
    
    # Saving the table to pickle
    joblib.dump(data,content_based_supervised_data + "/" + "rating_sample.pkl")

    reader = Reader(rating_scale=(1,10))
    data_sample = Dataset.load_from_df(rating_sample[['user_id', 'anime_id', 'rating']], reader)

    # Saving the table to pickle
    joblib.dump(data,content_based_supervised_data + "/" + "data_sample.pkl")

    return data_sample