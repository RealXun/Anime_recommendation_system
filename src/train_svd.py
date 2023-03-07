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
# Loads the best hyperparameters for the SVD algorithm that were obtained from grid search
print("\nLoading the best hyperparameters for the SVD algorithm that were obtained from grid search...")
gs = joblib.load(test_models + "/" + "SVD_best_params_test_model.pkl")
print("Loaded")

# Loads the dataset from a pickle file using joblib
print("\nLoading the dataset from a pickle file using joblib...")
data = joblib.load(processed_data + "/" + "data_reader_for_svd_model.pkl")    
print("Loaded")

# Splits the data into training and testing sets with a 80:20 ratio
print("\nSplitting the data into training and testing sets with an 80:20 ratio...")
trainset, testset = train_test_split(data, test_size=0.2)
print("Splitted")

# Creates an instance of the SVD algorithm with the best hyperparameters obtained from grid search
print("\nCreating an instance of the SVD algorithm with the best hyperparameters obtained from grid search...")
best_params = SVD(n_factors=gs['rmse']['n_factors'], 
                n_epochs=gs['rmse']['n_epochs'], 
                lr_all=gs['rmse']['lr_all'], 
                reg_all=gs['rmse']['reg_all'])
print("Created")

# Evaluate the model using cross-validation
print("\nEvaluating the model using cross-validation... (Take a sit, it might take up to 30min)")
cv_results = cross_validate(best_params, data, measures=['RMSE', 'MAE', 'MSE'], cv=5, verbose=True)
print("Evaluated")

# Print the average RMSE and MAE across all folds
print('Average RMSE Training:',  round(cv_results['test_rmse'].mean(), 2))
print('Average MAE Training:', round(cv_results['test_mae'].mean(), 2))
print('Average MSE Training:', round(cv_results['test_mse'].mean(), 2))


# Trains the SVD algorithm on the training set using the fit() method
print("\nTraining the SVD algorithm on the training set using the fit() method...")
best_params.fit(trainset)
print("Trained")

# Generates predictions for the test set using the trained model
print("\nGenerating predictions for the test set using the trained model...")
predictions = best_params.test(testset)
print("Predictions Done")

# Calculates the RMSE, MSE and MAE for the predictions
print("\nCalculating the RMSE, MSE and MAE for the predictions...")
rmse = round(accuracy.rmse(predictions, verbose=False), 2)
mse = round(accuracy.mse(predictions, verbose=False), 2)
mae = round(accuracy.mae(predictions, verbose=False), 2)

# Print the results of the RMSE, MSE and MAE for the predictions
print("\nPrinting the results of the RMSE, MSE and MAE for the predictions...")
print(f"RMSE Test: {rmse:.2f}")
print(f"MSE Test: {mse:.2f}")
print(f"MAE Test: {mae:.2f}") 

# Saves the trained model as a pickle file using joblib
print("\nSaving the trained model as a pickle file using joblib...")
joblib.dump(best_params,saved_models_folder + "/" + "SVD_new_model.pkl")
print("Model saved successfully!") 