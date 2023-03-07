# Standard library imports
import os # allows access to OS-dependent functionalities
import pandas as pd
from pathlib import Path
import pickle # serializing and deserializing a Python object structure.

# Third party libraries

import joblib # set of tools to provide lightweight pipelining in Python

# Unsupervised learner for implementing neighbor searches.
from sklearn.neighbors import NearestNeighbors

# deal with sparse data libraries
from scipy.sparse import csr_matrix # Returns a copy of column i of the matrix, as a (m x 1) CSR matrix (column vector).

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
# Saves the pivot table as a pickle file using joblib
df_pivot = joblib.load(processed_data + "/" + "pivot_user_based_unsupervised.pkl")

# Convert pivot table of user-item ratings to a sparse matrix in CSR format
matrix = csr_matrix(df_pivot.values)

# Create k-Nearest Neighbors model with 2 neighbors, Euclidean distance metric, brute force algorithm, and p-norm=2
model_knn = NearestNeighbors(n_neighbors=2, metric='euclidean', algorithm='brute', p=2)

# Fit k-Nearest Neighbors model on the user-item rating matrix
model_knn = model_knn.fit(matrix)

# Save the trained k-Nearest Neighbors model to a file using the pickle module
with open(saved_models_folder + "/" + "nearest_user_base_new_model.pkl", "wb") as f:
    pickle.dump(model_knn, f)
