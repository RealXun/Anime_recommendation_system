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
print('\nOpening CSV file called "anime.csv"...')
# CSV file called "anime.csv" from a directory called raw_data and returns the contents as a Pandas DataFrame
anime = pd.read_csv(raw_data + "/" + "anime.csv") 
print('Done.')

print('\nOpening CSV file called "rating.csv.zip"...')
# CSV file called "rating.csv.zip" from a directory called raw_data and returns the contents as a Pandas DataFrame
rating = pd.read_csv(raw_data + "/" + "rating.csv.zip") 
print('Done.')

print('\nCalling the cleaning functions...')
anime = cleaning.final_df()
print('Done.')

# Calling the cleaning functions
print('\nCalling the cleaning functions...')
anime_cleaned = cleaning.clean_anime_df(anime)
print('Done.')

print('\nMerging the given DataFrame with a rating DataFrame based on the anime_id column....')
merged_df = cleaning.merging(anime_cleaned)
print('Done.')

print('\nFiltering features dataframe (it might take a few moments but not enough for a coffee)....')
# This function preprocesses a merged dataframe by dropping users with no ratings, 
# saving the resulting dataframe to a pickle file, compressing it into a zip file, 
# and returning the resulting dataframe with users having at least 200 ratings.
features = cleaning.features_user_based_unsupervised(merged_df)
print('Done.')

print('\nCreating a pivot table with rows as anime titles, columns as user IDs, and the corresponding ratings as values (it might take a few moments but not enough for a coffee)....')
# creates a pivot table with rows as anime titles,columns as user IDs, and the corresponding ratings as values.
pivot_df = cleaning.create_pivot_table_unsupervised(features)
print('Done.')

# Convert pivot table of user-item ratings to a sparse matrix in CSR format
print("\nConverting pivot table to sparse matrix...")
matrix = csr_matrix(pivot_df.values)
print("Converted")

# Create k-Nearest Neighbors model with 2 neighbors, Euclidean distance metric, brute force algorithm, and p-norm=2
print("\nCreating k-NN model...")
model_knn = NearestNeighbors(n_neighbors=2, metric='euclidean', algorithm='brute', p=2)
print("Created")

# Fit k-Nearest Neighbors model on the user-item rating matrix
print("\nFitting k-NN model...")
model_knn = model_knn.fit(matrix)
print("Done")

# Save the trained k-Nearest Neighbors model to a file using the pickle module
print("\nSaving trained model...")
with open(saved_models_folder + "/" + "nearest_user_base_new_model.pkl", "wb") as f:
    pickle.dump(model_knn, f)
print("Model saved successfully!") 
