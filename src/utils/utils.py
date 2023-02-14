# data analysis and wrangling
import numpy as np
import pandas as pd
import warnings
import os
import re
import sys
import warnings
import joblib
import pickle
from fastparquet import write
from fastparquet import write 

from utils import utils
from sklearn.preprocessing import MinMaxScaler


from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy
from surprise.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors

pd.options.mode.chained_assignment = None  # default='warn'

#Preparing folder variables
#os.chdir(os.path.dirname(sys.path[0])) # This command makes the notebook the main path and can work in cascade.
#main_folder = sys.path[0]
#data_folder = (main_folder + "\data")
#saved_models_folder = (data_folder + "\saved_models")
#sounds_folder = (main_folder + "\sounds")
#saved_models = (main_folder + "\saved_models")
#processed_data = (data_folder + "\processed")
#sounds_folder = (main_folder + "\sounds")
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
data_folder = (PROJECT_ROOT + "\data")

saved_models_folder = (data_folder + "\saved_models")
raw_data = (data_folder + "\_raw")
processed_data = (data_folder + "\processed")
content_based_supervised_data = (data_folder + "\processed\content_based_supervised")

def anime():
    anime = pd.read_csv(data_folder + "/" + "anime.csv")
    return anime


def clean_supervised_content_based():

    anime = anime = pd.read_csv(data_folder + "/" + "anime.csv")

    # Droping null values
    anime_cleeaned = anime[anime["genre"].notna() & anime["type"].notna()] 

    # First, split the genre column by comma and expand the list so there is
    # a column for each genre. Now we have 13 columns, because the anime with
    # most genres tags has 13 tags
    genres = anime_cleeaned.genre.str.split(", ", expand=True)

    # Now we can get the list of unique genres. We "convert" the dataframe into
    # a single dimension array and take the unique values
    unique_genres = pd.Series(genres.values.ravel('K')).dropna().unique()

    # Getting the dummy variables will result in having a lot more columns
    # than unique genres
    dummies = pd.get_dummies(genres)

    # So we sum up the columns with the same genre to have a single column for
    # each genre
    for genre in unique_genres:
        anime_cleeaned[genre] = dummies.loc[:, dummies.columns.str.endswith(genre)].sum(axis=1)

    # Add the type dummies
    type_dummies = pd.get_dummies(anime_cleeaned["type"], prefix="", prefix_sep="")

    # Add the type dummies
    type_dummies = pd.get_dummies(anime_cleeaned["type"], prefix="", prefix_sep="")

    anime_cleeaned = pd.concat([anime_cleeaned, type_dummies], axis=1)
    anime_cleeaned = anime_cleeaned.drop(columns=["name", "type", "genre","anime_id"])

    anime_features = anime_cleeaned

    anime_features["episodes"] = anime_features["episodes"].map(lambda x:np.nan if x=="Unknown" else x)
    anime_features["episodes"].fillna(anime_features["episodes"].median(),inplace = True)
    anime_features["rating"] = anime_features["rating"].astype(float)
    anime_features["rating"].fillna(anime_features["rating"].median(),inplace = True)
    anime_features["members"] = anime_features["members"].astype(float)

    return anime_features



def model_NearestNeighbors(df):
    # Build and "train" the model
    neigh = NearestNeighbors(algorithm= 'auto', leaf_size= 30, metric= 'minkowski', n_neighbors= 100, p= 1, radius= 0.0).fit(df)
    distances, indices = neigh.kneighbors(df)

    joblib.dump(indices,saved_models_folder + "/" + "mode_based_content.pkl")

    return indices


def param_NearestNeighbors(df):

    parametros = { 'n_neighbors' : range(1,10),
               'radius' : np.linspace(0,1,11),
               'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
			   'leaf_size' : [30],
			   'metric' : ['minkowski','euclidean','manhattan'],
			   'p' : range(1,2)
			   }
    model_knn = GridSearchCV(NearestNeighbors(), parametros, scoring="accuracy",refit='precision_score', cv=2, n_jobs = -1).fit(df)

    return model_knn.best_params_

def get_index_from_name(name):
    anime = pd.read_csv(data_folder + "/" + "anime.csv")
    index = anime[anime["name"]==name].index.tolist()[0]
    return index


""" print_similar_query can search for similar animes both by id and by name. """

def print_similar_animes(ind,query,n):
    anime = pd.read_csv(data_folder + "/" + "anime.csv")
    found_id = get_index_from_name(query)
    array = ind[found_id][1:]
    indi = np.where(array==found_id)
    array = np.delete(array, indi)
    array = array[0:n]
    for id in array:
        print (anime[anime.index == id]['name'].values[0])