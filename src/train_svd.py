# Standard library imports
import os # allows access to OS-dependent functionalities
import pandas as pd
from pathlib import Path

# Third party libraries

import joblib # set of tools to provide lightweight pipelining in Python

# Python scikit for building and analyzing recommender systems that deal with explicit rating data
from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy

## scikit Cross validation iterators libraries
from sklearn.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split


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