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

print("\nLoading data, Filtering and sampling data, and converting it to a dataset object (Relax, it will not take much time)...")
#The code prepares data for different models by reading CSV files, filtering and sampling data, and converting it to a dataset object.
data = cleaning.prepare_for_different_models(1000000) # here we write the number of records we want to use, leaving it blank means using the whole dataset. Function in cleaning utils
print("Done")


# Splits the data into training and testing sets with a 80:20 ratio
print("\nSplitting the data into training and testing sets with an 80:20 ratio...")
trainset, testset = train_test_split(data, test_size=0.2)
print("Splitted")

# Creates an instance of the SVD algorithm with the best hyperparameters obtained from grid search
print("\nCreating an instance of the SVD algorithm with the best hyperparameters obtained from grid search...")
svd = SVD(n_factors=500, n_epochs=120, lr_all=0.01, reg_all=0.05)
print("Created")

# Evaluate the model using cross-validation
print("\nEvaluating the model using cross-validation... \n(Take a sit, it might take up to 30min. Feel free top grab a coffee)")
cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE', 'MSE','FCP'], cv=5, verbose=True)
print("Evaluated")

# Print the average RMSE and MAE across all folds
print('Average RMSE across all folds:', round(cv_results['test_rmse'].mean(), 2))
print('Average MAE across all folds:', round(cv_results['test_mae'].mean(), 2))
print('Average MSE across all folds:', round(cv_results['test_mse'].mean(), 2))
print('Average FCP across all folds:', cv_results['test_fcp'].mean())

print("\nFitting the model...")
# Trains the SVD algorithm on the training set using the fit() method
svd.fit(trainset)
print("Done")  

print("\nTesting the model...")
# Generates predictions for the test set using the trained model
predictions = svd.test(testset) 
print("Done")  

print("\nCalculating the RMSE, MSE and MAE for the predictions...")
# Calculates the RMSE, MSE and MAE for the predictions
rmse = round(accuracy.rmse(predictions, verbose=False), 3)
mse = round(accuracy.mse(predictions, verbose=False), 3)
mae = round(accuracy.mae(predictions, verbose=False), 3)
fcp = accuracy.fcp(predictions, verbose=False)
print("Done")  

# Print the results of the RMSE, MSE and MAE for the predictions
print(f"RMSE Test: {rmse:.3f}")
print(f"MSE Test: {mse:.3f}")
print(f"MAE Test: {mae:.3f}")
print(f"FCP Test: {fcp}")

# Saves the trained model as a pickle file using joblib
print("\nSaving the trained model as a pickle file using joblib...")
joblib.dump(svd,saved_models_folder + "/" + "SVD_new_model.pkl")
print("Model saved successfully!") 