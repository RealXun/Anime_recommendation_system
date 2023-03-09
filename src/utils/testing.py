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

## scikit Preprocessing data libraries
from sklearn.preprocessing import MinMaxScaler # Transform features by scaling each feature to a given range.
import joblib # set of tools to provide lightweight pipelining in Python
import glob

# Surprise libraries
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate,KFold
from surprise import Dataset, Reader, accuracy, SVD, SVDpp, BaselineOnly, CoClustering


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


def param_NearestNeighbors(df):
    '''
    The function param_NearestNeighbors uses GridSearchCV from scikit-learn 
    to perform a grid search over a range of hyperparameters for the 
    NearestNeighbors model. It takes a dataframe df as input and returns 
    the best hyperparameters found during the grid search. The hyperparameters 
    being searched over include n_neighbors, radius, algorithm, leaf_size, 
    metric, and p. The scoring metric being used is "accuracy" and the refit 
    parameter is set to "precision_score". cv=2 sets the number of cross-validation 
    folds to 2, and n_jobs=-1 sets the number of CPU cores used to parallelize 
    the search to be the maximum available.
    '''
    # Define dictionary of hyperparameters to test using GridSearchCV
    parametros = {
        'n_neighbors': range(1, 10),
        'radius': np.linspace(0, 1, 11),
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
        'p': range(1, 2)
    }

    # Create GridSearchCV object with NearestNeighbors algorithm
    # and hyperparameters defined in the parametros dictionary
    model_knn = GridSearchCV(NearestNeighbors(), parametros, scoring='accuracy', refit='precision_score', cv=2, n_jobs=-1).fit(df)

    # Return the best hyperparameters found by the grid search
    return model_knn.best_params_



#############################################################
#############################################################
#                                                           #
#--------- Unsupervised User  based recommendation ---------#
#                                                           #
#############################################################
#############################################################





##############################################################
##############################################################
#                                                            #
#----------- Supervised User based recommendation -----------#
#                                                            #
##############################################################
##############################################################

def svd_testing_parameters(size=None):
    '''
    defines a parameter grid for hyperparameter tuning in a svd filtering algorithm.
    Then create a GridSearchCV object with the SVD algorithm and a parameter grid consisting 
    of a range of hyperparameters. The GridSearchCV function then performs a grid search on 
    yhe parameter grid to find the best combination of hyperparameters that minimizes the 
    RMSE and MAE scores. The best RMSE and MAE scores and the corresponding parameters 
    are printed out.
    '''

    if size:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_1million.pkl")
    else:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_whole_data.pkl")

    # Define parameter grid for grid search
    param_grid = {'n_factors': [500], 
                'n_epochs': [120], 
                'lr_all': [0.01],
                'reg_all': [0.05]}
    
    # Create GridSearchCV object with SVD algorithmr
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae','mse','fcp'], cv=10)

    # Fit GridSearchCV object to data using multithreading and all available cores
    gs.fit(data)

    # Get results in a dataframe
    results_df = pd.DataFrame.from_dict(gs.cv_results)

    # Create a dataframe to store the best scores and parameters for each metric
    best_params_df = pd.DataFrame(columns=['dataset_size', 'model_name', 'metric', 'best_score', 'best_params'])

    # Print best scores and parameters for each metric
    for metric in ['rmse', 'mae', 'mse', 'fcp']:
        best_score_idx = results_df['rank_test_'+metric].idxmin()
        best_score = results_df.loc[best_score_idx]['mean_test_'+metric]
        best_params = results_df.loc[best_score_idx]['params']
        print(f"Best {metric.upper()} score:", best_score)
        print(f"Best parameters for {metric.upper()} score:", best_params)
    if size:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': '1million',
                                                                    'model_name': 'SVD',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "SVD_df_test_results_1million.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "SVD_best_test_results_1million.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "SVD_best_params_test_model_1million.pkl", compress = 1)
    else:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': 'whole data',
                                                                    'model_name': 'SVD',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "SVD_df_test_results_whole_data.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "SVD_best_test_results_whole_data.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "SVD_best_params_test_model_whole_data.pkl", compress = 1)


def baselineonly_testing_parameters(size=None):
    '''
    This code defines a function that performs a grid search with cross-validation using the BaselineOnly 
    algorithm from the Surprise library, to find the best hyperparameters for rating prediction on a given 
    dataset. It then extracts the best parameters and best scores for each metric (RMSE, MAE, MSE, and FCP), 
    saves the results to CSV files, and returns the best parameters. The function also includes an option to 
    use a smaller subset of the data (1 million ratings) or the entire dataset.
    '''

    # Parameters docs and value ranges:
    # http://surprise.readthedocs.io/en/stable/prediction_algorithms.html#baseline-estimates-configuration
    # http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf

    if size:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_1million.pkl")
    else:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_whole_data.pkl")

    # Define parameter grid for grid search
    param_grid = {'bsl_options': {'method': ['sgd', 'als'],
                                'reg': [0.02, 0.05],
                                'learning_rate': [0.005, 0.01],
                                'n_epochs': [10, 15],
                                'verbose': [True]},
                'verbose': [True]}
        
    # Create GridSearchCV object with BaselineOnly algorithm
    gs = GridSearchCV(BaselineOnly, param_grid, measures=['rmse', 'mae','mse','fcp'], cv=10)
    # Fit GridSearchCV object to data
    gs.fit(data)

    # Get results in a dataframe
    results_df = pd.DataFrame.from_dict(gs.cv_results)

    # Create a dataframe to store the best scores and parameters for each metric
    best_params_df = pd.DataFrame(columns=['dataset_size', 'model_name', 'metric', 'best_score', 'best_params'])

    # Print best scores and parameters for each metric
    for metric in ['rmse', 'mae', 'mse', 'fcp']:
        best_score_idx = results_df['rank_test_'+metric].idxmin()
        best_score = results_df.loc[best_score_idx]['mean_test_'+metric]
        best_params = results_df.loc[best_score_idx]['params']
        print(f"Best {metric.upper()} score:", best_score)
        print(f"Best parameters for {metric.upper()} score:", best_params)
    if size:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': '1million',
                                                                    'model_name': 'BaselineOnly',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "BaselineOnly_df_test_results_1million.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "BaselineOnly_best_test_results_1million.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "BaselineOnly_best_params_test_model_1million.pkl", compress = 1)
    else:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': 'whole data',
                                                                    'model_name': 'BaselineOnly',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "BaselineOnly_df_test_results_whole_data.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "BaselineOnly_best_test_results_whole_data.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "BaselineOnly_best_params_test_model_whole_data.pkl", compress = 1)



def coclustering_testing_parameters(size=None):
    '''
    This code defines a function that performs a grid search with cross-validation using the CoClustering 
    algorithm from the Surprise library, to find the best hyperparameters for rating prediction on a given 
    dataset. It then extracts the best parameters and best scores for each metric (RMSE, MAE, MSE, and FCP), 
    saves the results to CSV files, and returns the best parameters. The function also includes an option to 
    use a smaller subset of the data (1 million ratings) or the entire dataset.
    '''   

    if size:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_1million.pkl")
    else:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_whole_data.pkl")

    # Define parameter grid for grid search
    param_grid = {'n_cltr_u': [3, 5], 
                'n_cltr_i': [3, 5], 
                'n_epochs': [30, 40], 
                'verbose': [True]
                }
    # Create GridSearchCV object with CoClustering algorithm
    gs = GridSearchCV(CoClustering, param_grid, measures=['rmse', 'mae','mse','fcp'], cv=10)
    # Fit GridSearchCV object to data
    gs.fit(data)

    # Get results in a dataframe
    results_df = pd.DataFrame.from_dict(gs.cv_results)

    # Create a dataframe to store the best scores and parameters for each metric
    best_params_df = pd.DataFrame(columns=['dataset_size', 'model_name', 'metric', 'best_score', 'best_params'])

    # Print best scores and parameters for each metric
    for metric in ['rmse', 'mae', 'mse', 'fcp']:
        best_score_idx = results_df['rank_test_'+metric].idxmin()
        best_score = results_df.loc[best_score_idx]['mean_test_'+metric]
        best_params = results_df.loc[best_score_idx]['params']
        print(f"Best {metric.upper()} score:", best_score)
        print(f"Best parameters for {metric.upper()} score:", best_params)
    if size:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': '1million',
                                                                    'model_name': 'coclustering',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "coclustering_df_test_results_1million.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "coclustering_best_test_results_1million.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "coclustering_best_params_test_model_1million.pkl", compress = 1)
    else:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': 'whole data',
                                                                    'model_name': 'coclustering',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "coclustering_df_test_results_whole_data.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "coclustering_best_test_results_whole_data.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "coclustering_best_params_test_model_whole_data.pkl", compress = 1)


def svdpp_testing_parameters(size=None):
    '''
    defines a parameter grid for hyperparameter tuning in a svdpp filtering algorithm.
    Then create a GridSearchCV object with the SVDpp algorithm and a parameter grid consisting 
    of a range of hyperparameters. The GridSearchCV function then performs a grid search on 
    yhe parameter grid to find the best combination of hyperparameters that minimizes the 
    RMSE and MAE scores. The best RMSE and MAE scores and the corresponding parameters 
    are printed out.
    '''
    from surprise import SVDpp
    from surprise.model_selection import GridSearchCV

    if size:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_1million.pkl")
    else:
        # loading the table from pickle
        data = joblib.load(processed_data + "/" + "data_reader_for_different_models_whole_data.pkl")

    # Define parameter grid for grid search
    param_grid = {'n_factors': [300], 
                'n_epochs': [20], 
                'lr_all': [0.005],
                'reg_all': [0.1]}
    
    # Create GridSearchCV object with SVD algorithmr
    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae','mse','fcp'], cv=10, n_jobs=-1)

    # Fit GridSearchCV object to data using multithreading and all available cores
    gs.fit(data)

    # Get results in a dataframe
    results_df = pd.DataFrame.from_dict(gs.cv_results)

    # Create a dataframe to store the best scores and parameters for each metric
    best_params_df = pd.DataFrame(columns=['dataset_size', 'model_name', 'metric', 'best_score', 'best_params'])

    # Print best scores and parameters for each metric
    for metric in ['rmse', 'mae', 'mse', 'fcp']:
        best_score_idx = results_df['rank_test_'+metric].idxmin()
        best_score = results_df.loc[best_score_idx]['mean_test_'+metric]
        best_params = results_df.loc[best_score_idx]['params']
        print(f"Best {metric.upper()} score:", best_score)
        print(f"Best parameters for {metric.upper()} score:", best_params)
    if size:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': '1million',
                                                                    'model_name': 'svdpp',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "svdpp_df_test_results_1million.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "svdpp_best_test_results_1million.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "svdpp_best_params_test_model_1million.pkl", compress = 1)
    else:
        # Extract the best score and parameters for each metric
        for metric in ['rmse', 'mae', 'mse', 'fcp']:
            best_score_idx = results_df['rank_test_'+metric].idxmin()
            best_score = round(results_df.loc[best_score_idx]['mean_test_'+metric], 3)
            best_params = results_df.loc[best_score_idx]['params']
            best_params_df = pd.concat([best_params_df, pd.DataFrame({'dataset_size': 'whole data',
                                                                    'model_name': 'svdpp',
                                                                    'metric': metric.upper(), 
                                                                    'best_score': best_score, 
                                                                    'best_params': [best_params]})],
                                                                    ignore_index=True)
        # Save results to a CSV file
        results_df.to_csv(test_models + "/" + "svdpp_df_test_results_whole_data.csv", index=False)

        # Save the best scores and parameters to a CSV file
        best_params_df.to_csv(test_models + "/" + "svdpp_best_test_results_whole_data.csv", index=False)

        # Save best parameters
        joblib.dump(gs.best_params,test_models + "/" + "svdpp_best_params_test_model_whole_data.pkl", compress = 1)