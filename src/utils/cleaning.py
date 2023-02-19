# Standard library imports
import os # allows access to OS-dependent functionalities
import re #  regular expression matching operations similar to those found in Perl
import sys # to manipulate different parts of the Python runtime environment
import warnings # is used to display the message Warning
import pickle # serializing and deserializing a Python object structure.
from zipfile import ZipFile

# Third party libraries
from fastparquet import write # parquet format, aiming integrate into python-based big data work-flows
from fuzzywuzzy import fuzz # used for string matching

import numpy as np # functions for working in domain of linear algebra, fourier transform, matrices and arrays
import pandas as pd # data analysis and manipulation tool
import joblib # set of tools to provide lightweight pipelining in Python

# Utils libraries
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
data_folder = (PROJECT_ROOT + "/" + "data")

saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")
content_based_supervised_data = (data_folder + "/" + "processed" + "/" + "content_based_supervised")

############################################################
############################################################
#                                                          #
#--------------------- For all models ---------------------#
#                                                          #
############################################################
############################################################

'''

'''
def anime():
    anime = pd.read_csv(raw_data + "/" + "anime.csv")
    return anime


'''

'''
def rating():
    rating = pd.read_csv(raw_data + "/" + "rating.csv.zip")
    return rating

'''
A marge of anime.csv with anime_2023_02_15_00_49_41.csv to get more information like cover or japanese tittle
'''
def final_df():
    anime_no_cleaned = pd.read_csv(raw_data + "/" + "anime.csv")# load anime df
    anime_new = pd.read_csv(raw_data + "/" + "anime_2023_02_15_00_08_28.csv",sep=";")# load anime df
    anime_final = pd.merge(anime_new[["anime_id",'English_Title',"Japanses_Title","Source","Duration","Rating","Score","Rank","synopsis","Cover"]]\
        , anime_no_cleaned[["anime_id",'name','genre',"type","episodes","members"]]\
        , on = "anime_id")
    anime_final = anime_final[['anime_id',"name", 'English_Title', 'Japanses_Title',"genre", 'type', 'Source', 'Duration', 'episodes', 'Rating', 'Score',"Rank", 'members', 'synopsis',"Cover"]]
    anime_final= anime_final.rename(columns=str.lower)
    anime_final.to_csv(processed_data + "/" + "anime_final.csv", index=False)
    return anime_final

'''

'''
def clean_anime_df(anime):
    anime_cleaned = anime

    anime_cleaned['name'] = anime_cleaned['name'].str.replace('\W', ' ', regex=True)
    
    # Cambiamos a minúsculas todos los nombre de animes
    anime_cleaned['name'] = anime_cleaned['name'].str.lower()

    anime_cleaned["episodes"] = anime_cleaned["episodes"].map(lambda x:np.nan if x=="Unknown" else x)
    anime_cleaned["episodes"].fillna(anime_cleaned["episodes"].median(),inplace = True)
    anime_cleaned["score"] = anime_cleaned["score"].astype(float)
    anime_cleaned["score"].fillna(anime_cleaned["score"].median(),inplace = True)
    anime_cleaned["members"] = anime_cleaned["members"].astype(float)

    anime_to_compare = anime_cleaned
    #joblib.dump(anime_to_compare,processed_data + "/" + "_anime_to_compare_with_name.pkl")
    anime_cleaned.to_csv(processed_data + "/" + "_anime_to_compare_with_name.csv", index=False)
    # Droping null values
    anime_cleaned = anime_cleaned[anime_cleaned["genre"].notna() & anime_cleaned["type"].notna()]
    #joblib.dump(anime_cleaned,processed_data + "/" + "anime_cleaned.pkl")
    anime_cleaned.to_csv(processed_data + "/" + "anime_cleaned.csv", index=False)
    return anime_cleaned

############################################################
############################################################
#                                                          #
#----- Unsupervised User content based recommendation -----#
#                                                          #
############################################################
############################################################


'''

'''
def prepare_supervised_content_based(anime_cleaned):

    # First, split the genre column by comma and expand the list so there is
    # a column for each genre. Now we have 13 columns, because the anime with
    # most genres tags has 13 tags
    genres = anime_cleaned.genre.str.split(", ", expand=True)

    # Now we can get the list of unique genres. We "convert" the dataframe into
    # a single dimension array and take the unique values
    unique_genres = pd.Series(genres.values.ravel('K')).dropna().unique()

    # Getting the dummy variables will result in having a lot more columns
    # than unique genres
    dummies = pd.get_dummies(genres)

    # So we sum up the columns with the same genre to have a single column for
    # each genre
    for genre in unique_genres:
        anime_cleaned[genre] = dummies.loc[:, dummies.columns.str.endswith(genre)].sum(axis=1)

    # Add the type dummies
    type_dummies = pd.get_dummies(anime_cleaned["type"], prefix="", prefix_sep="")

    # Add the type dummies
    type_dummies = pd.get_dummies(anime_cleaned["type"], prefix="", prefix_sep="")

    anime_cleaned = pd.concat([anime_cleaned, type_dummies], axis=1)
    anime_cleaned = anime_cleaned.drop(columns=["name", "type", "genre","anime_id","english_title","japanses_title","source","duration",\
        "episodes","rating","score","rank","members","synopsis","cover"])
    
    anime_features = anime_cleaned
    anime_features.reset_index(drop=True)

    #joblib.dump(anime_features,processed_data + "/" + "anime_features.pkl")
    anime_cleaned.to_csv(processed_data + "/" + "anime_features.csv", index=False)
    return anime_features


#############################################################
#############################################################
#                                                           #
#--------- Unsupervised User  based recommendation ---------#
#                                                           #
#############################################################
#############################################################


'''
'''
def merging(df):
    ratingdf = cleaning.rating()
    # Añadimos suffixes for ratingdf ya que en los dos df la columna rating tiene el mismo nombre
    merged_df=pd.merge(df,ratingdf,on='anime_id',suffixes= ['', '_user']) 

    # Cambiamos un par de nombres de columnas
    merged_df = merged_df.rename(columns={'name': 'name', 'rating_user': 'user_rating'})
#
    return merged_df


'''
'''
def features_user_based_unsupervised(df_merged):

    # A user who hasn't given any ratings (-1) has added no value to the engine. So let's drop it.
    features=df_merged.copy()
    features["user_rating"].replace({-1: np.nan}, inplace=True)
    features = features.dropna(axis = 0, how ='any')

    # There are users who has rated only once. So we should think if we want to consider only users with a minimin ratings as threshold value. Let's say 50.
    counts = features['user_id'].value_counts()
    features = features[features['user_id'].isin(counts[counts >= 200].index)]

    # Saving the pivot table to pickle
    joblib.dump(features,processed_data + "/" + "features_user_based_unsupervised.pkl")

    return features


'''
'''
def create_pivot_table_unsupervised(df_features):
    # This pivot table consists of rows as title and columns as user id, this will help us to create sparse matrix which can be very helpful in finding the cosine similarity
    pivot_df=df_features.pivot_table(index='name',columns='user_id',values='user_rating').fillna(0)

    # Saving the table to pickle
    joblib.dump(pivot_df,processed_data + "/" + "pivot_user_based_unsupervised.pkl")

    #import zipfile as ZipFile
    #import zipfile

    ## zipping the file
    #with zipfile.ZipFile(processed_data + "/" + 'pivot_user_based_unsupervised.zip',"w", zipfile.ZIP_DEFLATED) as zipf:
    #    zipf.write(processed_data + "/" + "pivot_user_based_unsupervised.pkl")
    #    zipf.close()

    import zipfile as ZipFile
    import zipfile

    dir, base_filename = os.path.split(processed_data + "/" + "pivot_user_based_unsupervised.pkl")
    os.chdir(dir)
    zip = zipfile.ZipFile('pivot_user_based_unsupervised.zip',"w", zipfile.ZIP_DEFLATED)
    zip.write(base_filename)
    zip.close()

    to_find_index=pivot_df.reset_index()
    to_find_index = to_find_index[["name"]]

    # Saving the table to pickle
    #joblib.dump(to_find_index,processed_data + "/" + "_to_find_index_user_based_unsupervised.pkl")
    to_find_index.to_csv(processed_data + "/" + "_to_find_index_user_based_unsupervised.csv")

    return pivot_df


##############################################################
##############################################################
#                                                            #
#----------- Supervised User based recommendation -----------#
#                                                            #
##############################################################
##############################################################

# Cleaning the data
def supervised_rating_cleaning(rating):
    ratingdf = rating[rating.rating>0]
    ratingdf = ratingdf.reset_index()
    ratingdf.drop('index', axis=1,inplace=True)
    return ratingdf
