# Standard library imports
import os # allows access to OS-dependent functionalities
import re #  regular expression matching operations similar to those found in Perl
import sys # to manipulate different parts of the Python runtime environment
import warnings # is used to display the message Warning
import pickle # serializing and deserializing a Python object structure.
from zipfile import ZipFile
import pandas as pd


# Third party libraries
from fastparquet import write # parquet format, aiming integrate into python-based big data work-flows
from fuzzywuzzy import fuzz # used for string matching

import numpy as np # functions for working in domain of linear algebra, fourier transform, matrices and arrays
import pandas as pd # data analysis and manipulation tool
import joblib # set of tools to provide lightweight pipelining in Python

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from surprise import Dataset, Reader, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering


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
This function reads a CSV file called "anime.csv" from a 
directory called raw_data and returns the contents as a Pandas DataFrame.
'''
def anime():
    anime = pd.read_csv(raw_data + "/" + "anime.csv")
    return anime


'''
This function reads a CSV file called "rating.csv" from a 
directory called raw_data and returns the contents as a Pandas DataFrame.
'''
def rating():
    rating = pd.read_csv(raw_data + "/" + "rating.csv.zip")
    return rating

'''
This function merges the two dataframes of anime data, reorders and selects columns, 
renames the columns to lowercase, and saves the resulting dataframe to a CSV file. 
The merged and cleaned dataframe is returned as the output.
In other words, we get more information like to get more information like cover or japanese tittle
'''
def final_df():
    # Load the original anime dataframe
    anime_no_cleaned = pd.read_csv(raw_data + "/" + "anime.csv")
    
    # Load the updated anime dataframe
    anime_new = pd.read_csv(raw_data + "/" + "anime_2023_02_15_00_08_28.csv", sep=";")
    
    # Merge the two dataframes on the anime_id column
    anime_final = pd.merge(
        anime_new[["anime_id", "English_Title", "Japanses_Title", "Source", "Duration", "Rating", "Score", "Rank", "synopsis", "Cover"]],
        anime_no_cleaned[["anime_id", "name", "genre", "type", "episodes", "members"]],
        on="anime_id"
    )
    
    # Reorder and select columns
    anime_final = anime_final[["anime_id", "name", "English_Title", "Japanses_Title", "genre", "type", "Source", "Duration", "episodes", "Rating", "Score", "Rank", "members", "synopsis", "Cover"]]
    
    # Rename columns to lower case
    anime_final = anime_final.rename(columns=str.lower)
    
    # Save the final dataframe to a CSV file in the processed data directory
    anime_final.to_csv(processed_data + "/" + "anime_final.csv", index=False)
    
    # Return the final dataframe
    return anime_final



'''
The function takes a pandas dataframe containing anime data and 
fills in missing values in the 'source' column using a Decision 
Tree Classifier based on the 'episodes' and 'type' columns. The 
'type' column is converted to categorical data using get_dummies 
before fitting the model. The function returns the original 
dataframe with missing values filled in and the model accuracy score.
'''

def predict_source(anime_cleaned):
    # change unknown values to NaN
    anime_cleaned['source'].replace('Unknown', pd.NA, inplace=True)

    # fill missing values in the 'episodes' column with 0
    anime_cleaned['episodes'].fillna(0, inplace=True)

    # create dummy variables for the 'type' column
    anime_cleaned = pd.get_dummies(anime_cleaned, columns=['type'])

    # split the data into training and validation sets
    train_data = anime_cleaned[anime_cleaned['source'].notna()]
    validation_data = anime_cleaned[anime_cleaned['source'].isna()]

    # create the decision tree classifier
    model = DecisionTreeClassifier()

    # train the model using the training data
    model.fit(train_data[['episodes', 'type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']],
              train_data['source'])

    # predict the 'source' values for the validation data
    predicted_sources = model.predict(validation_data[['episodes', 'type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']])

    # fill the 'NaN' 'source' values in the original DataFrame with the predicted values
    anime_cleaned.loc[anime_cleaned['source'].isna(), 'source'] = predicted_sources

    # undo the get_dummies() operation to convert the one-hot encoded 'type' columns back to a single categorical column
    anime_cleaned['type'] = anime_cleaned[['type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']].idxmax(axis=1).str.replace('type_', '')
    anime_cleaned.drop(columns=['type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV'], inplace=True)
    # calculate the accuracy of the model
    accuracy = accuracy_score(train_data['source'], model.predict(train_data[['episodes', 'type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']]))

    return anime_cleaned



'''
The function clean_anime_df() takes an anime dataframe as input and performs several 
cleaning and preprocessing steps, such as removing special characters from anime names, 
converting all names to lowercase, filling missing values for "episodes" and "score" 
columns with their median, dropping rows with null values for "genre" or "type" columns, 
and saving the cleaned dataframe to a CSV file. The cleaned dataframe is also returned as output.

'''
def clean_anime_df(anime):
    '''The function clean_anime_df() takes an anime dataframe as input and performs several 
    cleaning and preprocessing steps, such as removing special characters from anime names, 
    converting all names to lowercase, filling missing values for "episodes" and "score" 
    columns with their median, dropping rows with null values for "genre" or "type" columns, 
    and saving the cleaned dataframe to a CSV file. The cleaned dataframe is also returned as output.'''

    anime_cleaned = anime

    anime_cleaned['name'] = anime_cleaned['name'].str.replace('\W', ' ', regex=True)
    
    anime_cleaned['name'] = anime_cleaned['name'].str.lower()

    anime_cleaned["episodes"] = anime_cleaned["episodes"].map(lambda x:np.nan if x=="Unknown" else x)

    anime_cleaned["episodes"].fillna(anime_cleaned["episodes"].median(),inplace = True)

    anime_cleaned["score"] = anime_cleaned["score"].astype(float)

    anime_cleaned["score"].fillna(anime_cleaned["score"].median(),inplace = True)

    anime_cleaned["members"] = anime_cleaned["members"].astype(float)

    anime_cleaned.to_csv(processed_data + "/" + "_anime_to_compare_with_name.csv", index=False)

    anime_cleaned['genre'] = anime_cleaned['genre'].fillna(anime_cleaned['genre'].mode()[0])

    anime_cleaned['type'] = anime_cleaned['type'].fillna(anime_cleaned['type'].mode()[0])

    anime_cleaned = predict_source(anime_cleaned)

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
This function prepares the content-based features for a supervised 
learning model. It first splits the genres into separate columns and 
gets unique genres. It then creates dummy variables for genres and 
type, and sum up the columns for the same genre to have a single 
column for each genre. Finally, it drops irrelevant columns and saves 
the resulting dataframe to a CSV file. The function returns the resulting dataframe.
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

    anime_cleaned = pd.concat([anime_cleaned, type_dummies], axis=1)
    anime_cleaned = anime_cleaned.drop(columns=["name", "type", "genre","anime_id","english_title","japanses_title","source","duration","rating","rank","synopsis","cover"])
    
    anime_features = anime_cleaned
    anime_features.reset_index(drop=True)

    #joblib.dump(anime_features,processed_data + "/" + "anime_features.pkl")
    anime_cleaned.to_csv(processed_data + "/" + "anime_features.csv", index=False)
    return anime_features


#############################################################
#############################################################
#                                                           #
#------ Unsupervised User rating based recommendation ------#
#                                                           #
#############################################################
#############################################################


'''
This function merges the given DataFrame with a rating DataFrame 
based on the anime_id column. It then renames the 'rating_user' 
column to 'user_rating' and returns the merged DataFrame.
'''
def merging(df):
    ratingdf = rating()
    # Añadimos suffixes for ratingdf ya que en los dos df la columna rating tiene el mismo nombre
    merged_df=pd.merge(df,ratingdf,on='anime_id',suffixes= ['', '_user']) 

    # Cambiamos un par de nombres de columnas
    merged_df = merged_df.rename(columns={'name': 'name', 'rating_user': 'user_rating'})
#
    return merged_df


'''
This function takes in a merged dataframe, preprocesses the data to drop users 
who have not given any ratings and users who have given fewer ratings than a 
specified threshold value, and saves the resulting pivot table to a pickle file. 
It then compresses the pickle file into a zip file and returns the resulting pivot table.
'''
def features_user_based_unsupervised(df_merged):
    # A user who hasn't given any ratings (-1) has added no value to the engine. So let's drop it.
    features=df_merged.copy()
    features["user_rating"].replace({-1: np.nan}, inplace=True)
    features = features.dropna(axis = 0, how ='any')
    # Drop rows with NaN values (user has not given any ratings)
    
    # There are users who has rated only once. So we should think if we want to consider only users with a minimin ratings as threshold value. Let's say 50.
    counts = features['user_id'].value_counts()
    features = features[features['user_id'].isin(counts[counts >= 200].index)]
    # Only consider users with at least 200 ratings
    
    # Saving the pivot table to pickle
    joblib.dump(features,processed_data + "/" + "features_user_based_unsupervised.pkl")
    
    # Create a zip file for the saved pickle file
    import zipfile as ZipFile
    import zipfile
    dir, base_filename = os.path.split(processed_data + "/" + "features_user_based_unsupervised.pkl")
    os.chdir(dir)
    zip = zipfile.ZipFile('features_user_based_unsupervised.zip',"w", zipfile.ZIP_DEFLATED)
    zip.write(base_filename)
    zip.close()
    
    # Return the cleaned and filtered features dataframe
    return features



'''
The function create_pivot_table_unsupervised creates a pivot table with rows as anime titles, 
columns as user IDs, and the corresponding ratings as values. The pivot table is then saved 
to a pickle file and zipped. The function also saves a separate file containing only the 
anime titles. Finally, the pivot table is returned.
'''
def create_pivot_table_unsupervised(df_features):
    # This pivot table consists of rows as title and columns as user id, this will help us to create sparse matrix which can be very helpful in finding the cosine similarity
    pivot_df=df_features.pivot_table(index='name',columns='user_id',values='user_rating').fillna(0)

    # Saving the table to pickle
    joblib.dump(pivot_df,processed_data + "/" + "pivot_user_based_unsupervised.pkl")

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
'''
This function takes a pandas DataFrame named "rating" and removes all 
the rows where the "rating" column has a value of 0 or less, and 
then resets the index of the resulting DataFrame. It returns the 
cleaned DataFrame.
'''
# Cleaning the data
def supervised_rating_cleaning(rating):
    ratingdf = rating[rating.rating>0]
    ratingdf = ratingdf.reset_index()
    ratingdf.drop('index', axis=1,inplace=True)
    return ratingdf



'''
The code reads two CSV files (anime.csv and rating.csv.zip) and loads them into dataframes. 
Then it creates a subset of the rating dataframe containing only rows where the rating is 
greater than 0 and removes the index column. Next, it samples a subset of the data with 
a specified size, grouped by the rating column.
'''
def supervised_prepare_training():
    # Load 'anime.csv' file into a pandas DataFrame object called 'anime'
    anime = pd.read_csv(raw_data + "/" + "anime.csv")

    # Load 'rating.csv.zip' file into a pandas DataFrame object called 'rating'
    rating = pd.read_csv(raw_data + "/" + "rating.csv.zip")

    # Create a new DataFrame 'anime_mapping' that is a copy of the 'anime' DataFrame and remove the 'episodes', 'members', and 'rating' columns
    anime_mapping = anime.copy()
    anime_mapping.drop(['episodes','members','rating'],axis=1, inplace=True)

    # Filter out all ratings less than or equal to 0 and reset the index of the DataFrame
    ratingdf = rating[rating.rating>0]
    ratingdf = ratingdf.reset_index()

    # Drop the 'index' column and update the DataFrame in-place
    ratingdf.drop('index', axis=1, inplace=True)

    # Get the shape of the DataFrame 'ratingdf'
    ratingdf.shape

    # Set the size to 100,000 and sample from the 'ratingdf' DataFrame based on the proportion of ratings for each score
    size = 100000

    # This will make sure that the sampled data has roughly the same proportion of ratings for each score as the original data
    ratingdf_sample = ratingdf.groupby("rating", group_keys=False).apply(lambda x: x.sample(int(np.rint(size*len(x)/len(ratingdf))))).sample(frac=1).reset_index(drop=True)

    # Create a new 'Reader' object with the rating scale set to a range between 1 and 10
    reader = Reader(rating_scale=(1,10))

    # Load the sampled data into a 'Dataset' object using the 'load_from_df' method and the 'reader' object
    data = Dataset.load_from_df(ratingdf_sample[['user_id', 'anime_id', 'rating']], reader)

    # Saving the table to pickle
    joblib.dump(data,processed_data + "/" + "data_reader_sample.pkl")

    return data