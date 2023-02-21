# data analysis and wrangling
import pandas as pd
import numpy as np
import warnings
import os
import re
import sys
import warnings
import joblib
import pickle
from fastparquet import write
from fuzzywuzzy import fuzz
from pathlib import Path
import zipfile
import shutil

from sklearn.preprocessing import MinMaxScaler


from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy
from surprise.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors

pd.options.mode.chained_assignment = None  # default='warn'


PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
data_folder = (PROJECT_ROOT + "/" + "data")

saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")
content_based_supervised_data = (data_folder + "/" + "processed" + "/" + "content_based_supervised")



#shutil.unpack_archive(processed_data + "/" + "features_user_based_unsupervised.zip",processed_data)
#shutil.unpack_archive(processed_data + "/" + "pivot_user_based_unsupervised.zip",processed_data)

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



def names_unique():
    '''
    Return a list of unique names in the column 'english_title'
    '''
    anime = pd.read_csv(raw_data + "/" + "anime.csv")
    names = anime['english_title'].unique().tolist()
    return names



def all_anime_dict():
    '''
    The code creates a dictionary from a Pandas DataFrame called "anime", 
    where the keys of the dictionary are the column names of the DataFrame 
    and the values of the dictionary are the corresponding 
    Pandas Series (i.e., column) of the DataFrame.
    '''
    anime_dict = anime().to_dict('records')
    return anime_dict




def from_index_to_title(index,df):
    '''
    Function to return the anime name that mtches de index number
    '''
    anime = df
    return anime[anime.index == index]['name'].values[0]



def from_title_to_index(title,df):
    '''
    Function to return the matched index number of the anime name
    '''
    anime = df
    return anime[anime["name"]==title].index.values[0]



def match_the_score(a,b):
    '''
    Function to find the closest title, It uses Levenshtein Distance to calculate the differences between sequences
    '''
    return fuzz.ratio(a,b)




def finding_the_closest_title(title,df):
    '''
    Function that takes in a string title and a pandas DataFrame df as input arguments, 
    and returns a tuple containing the closest matching title to the input title 
    and the Levenshtein distance score between the closest title and the input title.
    in other words, the function returns the most similar title to the name a user typed
    '''
    # This function takes a string `title` and a pandas DataFrame `df` as input arguments.

    # Create a new variable `anime` to hold the DataFrame `df` for readability.
    anime = df
   
    # Calculate the Levenshtein distance between each title in the 'name' column of the DataFrame and the input `title`.
    # The `match_the_score` function is used to calculate the distance score.
    # The `enumerate` function adds an index number to each distance score.
    levenshtein_scores = list(enumerate(anime['name'].apply(match_the_score, b=title)))

    # Sort the list of (index, distance score) tuples in descending order by the distance score.
    sorted_levenshtein_scores = sorted(levenshtein_scores, key=lambda x: x[1], reverse=True)

    # Get the closest matching title to the input `title` by using the index of the highest scoring match.
    # The `from_index_to_title` function is used to return the title string from the DataFrame given an index.
    closest_title = from_index_to_title(sorted_levenshtein_scores[0][0],anime)

    # Get the Levenshtein distance score of the closest matching title.
    distance_score = sorted_levenshtein_scores[0][1]

    # Return a tuple containing the closest matching title and its Levenshtein distance score.
    return closest_title, distance_score




def filtering_or(df, genres, types):
    '''
    The code defines a function "filtering_or" that filters a pandas dataframe based on user-defined 
    genres and types using an "OR" method. The function allows the user to select one or all possible 
    genres and types and returns a filtered dataframe with the selected genres and types. 
    The function also splits the genre and type columns and explodes them to account for multiple entries.
    '''

    # Make a copy of the input DataFrame
    filtered_df = df.copy()
    
    # Split the genre column into a list of genres
    filtered_df['genre'] = filtered_df['genre'].str.split(', ')
    
    # Explode the genre column to create a new row for each genre in the list
    filtered_df = filtered_df.explode('genre')
    
    # If genres are specified and 'ALL' is not one of them, filter the DataFrame to keep only rows where the genre is in the specified list
    if genres and 'ALL' not in genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(genres)]
        
    # If types are specified and 'ALL' is not one of them, filter the DataFrame to keep only rows where the type is in the specified list
    if types and 'ALL' not in types:
        filtered_df = filtered_df[filtered_df['type'].apply(lambda type: type in types) if isinstance(filtered_df['type'].iloc[0], str) else False]
    
    # If both genres and types are specified
    if genres and types:
        # If 'ALL' is in the genres list, set genres to be all the unique genres in the filtered DataFrame
        if 'ALL' in genres:
            genres = filtered_df['genre'].unique()
        # If 'ALL' is in the types list, set types to be all the unique types in the filtered DataFrame
        if 'ALL' in types:
            types = filtered_df['type'].unique()

        # Filter the DataFrame to keep only rows where the genre is in the genres list AND the type is in the types list
        filtered_df = filtered_df[(filtered_df['genre'].isin(genres)) & (filtered_df['type'].isin(types))]
    
    # Return the filtered DataFrame
    return filtered_df




def filtering_and(df, genres, types):
    '''
    This function takes a DataFrame df, a list of genres, and a list of types as input arguments. 
    The function first creates a boolean mask genre_mask by applying a lambda function to 
    the 'genre' column of the DataFrame. The lambda function checks if the value is a 
    string using isinstance(x, str) and if all genres in the genres list are present 
    in the string, which is split by comma and space using x.split(', '). 
    The all() function returns True if all genres in the genres list are present 
    in the string. The resulting genre_mask will be True for rows where the genre 
    column contains all of the genres in the genres list.

    Then the function creates another boolean mask type_mask by using the isin() 
    method to check if each value in the 'type' column of the DataFrame is in the types list.

    Finally, the function applies both masks to the DataFrame df using the & operator 
    to create a new DataFrame filtered_df that includes only rows where both m
    asks are True. The function returns the filtered DataFrame.
    '''
    # This function takes a DataFrame `df`, a list of `genres`, and a list of `types` as input arguments.

    # Create a boolean mask that filters rows where the genre column contains all of the genres in the `genres` list.
    genre_mask = df['genre'].apply(lambda x: isinstance(x, str) and all([genre in x.split(', ') for genre in genres]))

    # Create a boolean mask that filters rows where the type column is in the `types` list.
    type_mask = df['type'].isin(types)

    # Apply both masks to the DataFrame `df` and create a new DataFrame `filtered_df` that includes only rows where both masks are True.
    filtered_df = df[genre_mask & type_mask]

    # Return the filtered DataFrame.
    return filtered_df




    
def create_dict(names, gen, typ, method, n=200):
    '''
    The create_dict() function takes in four arguments - names (list of anime names to search for), 
    gen (list of genres to filter by), typ (list of anime types to filter by), 
    method (string indicating whether to filter by "or" or "and"), 
    and an optional n parameter indicating the maximum number of results to return. 
    It reads in a pre-processed anime DataFrame, filters it based on the input criteria, 
    and returns a dictionary of the resulting rows. If there are no matches, 
    it returns a string indicating it.
    '''
    # This function takes in a list of anime titles `names`, lists of `gen`res and `typ`es, a filtering method `method`, and an optional number of results `n`.
    
    # Load the anime dataframe from a CSV file using pandas.
    anime = pd.read_csv(processed_data + "/" + "_anime_to_compare_with_name.csv")
    
    # Filter the anime dataframe to only include titles that match those in the input list `names`.
    final_df = anime[anime['name'].isin(names)]

    # Remove the 'anime_id' and 'members' columns from the resulting dataframe.
    final_df = final_df.drop(columns=["anime_id", "members"])

    # Reset the index of the resulting dataframe.
    blankIndex=[''] * len(final_df)
    final_df.index=blankIndex

    # Apply a filtering method based on the input `method`.
    if method == 'or': # If 'or', use the `filtering_or()` function to filter the dataframe.
        print("or")
        final_df = filtering_or(final_df, gen, typ)
    elif method == 'and':# If 'and', use the `filtering_and()` function to filter the dataframe.
        print("and")
        final_df = filtering_and(final_df, gen, typ)
    else: # If `method` is neither 'or' nor 'and', raise a ValueError.
        raise ValueError("Invalid filter type. Expected 'or' or 'and'.")

    final_df = final_df.drop_duplicates(subset=["name"])# Drop any duplicate titles from the resulting dataframe.
    final_df = final_df.head(n)# Limit the resulting dataframe to the first `n` rows.

    if final_df.empty:# If the resulting dataframe is empty, print an error message and return None.
        sentence = print('WOW!!!! Sorry, there is no matches for the anime and options selected! \n Try again, you might have mroe luck')
        return sentence
    else:# Otherwise, convert the resulting dataframe to a dictionary and return the dictionary.
        final_dict = final_df.to_dict('records')
        return final_dict



############################################################
############################################################
#                                                          #
#----- Unsupervised User content based recommendation -----#
#                                                          #
############################################################
############################################################


def print_similar_animes(query):
    '''
    This function takes a user input anime name query and returns a list of recommended anime similar to the query.
    It uses a pre-trained model and a dataset of anime information to find recommendations. 
    If the user query has any misspelling, the function tries to find the closest match to 
    the query and provides recommendations based on that.
    '''
    ind = joblib.load(saved_models_folder + "/" + "kNearest_user_content_new_model.pkl")
    anime = pd.read_csv(processed_data + "/" + "_anime_to_compare_with_name.csv")
    closest_title, distance_score = finding_the_closest_title(query,anime)
       
    if distance_score == 100:
        names = []
        errors = []
        print('These are the recommendations for similar animes to '+'\033[1m'+str(query)+'\033[0m'+'','\n')
        found_id = from_title_to_index(query,anime) 
        array = ind[found_id][1:] 
        indi = np.where(array==found_id) 
        array = np.delete(array, indi) 
        for id in array:
            try :
                names.append(anime[anime.index == id]['name'].values[0])
            except IndexError :
                errors.append(id)
        return names

   # When a user makes misspellings    
    else:
        names = []
        errors = []
        print('I guess you misspelled the name\n Are you looking similitudes for the anime named '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n' + 'Here are the recommendations:')
        found_id = from_title_to_index(closest_title,anime) 
        array = ind[found_id][1:] 
        indi = np.where(array==found_id)
        array = np.delete(array, indi) 
        for id in array:
            try :
                names.append(anime[anime.index == id]['name'].values[0])
            except IndexError :
                errors.append(id)
        return names




#############################################################
#############################################################
#                                                           #
#--------- Unsupervised user-based recommendation ----------#
#                                                           #
#############################################################
#############################################################


def reco(name, n, df):
    '''
    This function returns a list of recommended animes based on user-based collaborative filtering. 
    It loads a pre-trained KNN model and a pivot table containing user-anime ratings. 
    It then finds the index of the anime provided by the user and calculates the distances 
    and indices of the most similar animes. Finally, it returns a list of recommended animes. 
    '''
    # Load the trained KNN model for user-based unsupervised learning.
    model_knn = joblib.load(saved_models_folder + "/" + "nearest_user_base_new_model.pkl")

    #shutil.unpack_archive(processed_data + "/" + "pivot_user_based_unsupervised.zip",processed_data)
    #pivot_df = pd.read_csv(processed_data + "/" + "pivot_user_based_unsupervised.zip")# load anime df
    
    # Load the pivot table which stores the user rating data.
    pivot_df = joblib.load(processed_data + "/" + "pivot_user_based_unsupervised.pkl")

    # Get the index of the anime given the name of the anime.
    indl = from_title_to_index(name, df)

    # Get the n nearest neighbors (anime recommendations) of the given anime.
    distances, indices = model_knn.kneighbors(pivot_df.iloc[indl,:].values.reshape(1, -1), n_neighbors = n+1)

    # Store the names of the n nearest neighbors in a list.
    names_list = []
    for i in range(1, n+1):
        # If no recommendations are found, print a message to the user.
        if i == 0:
            print('WOW!!!! Sorry, there is no matches for the anime and options selected!\nTry again, you might have more luck')
        else:
            names_list.append(pivot_df.index[indices.flatten()[i]])
            #print('{0}: {1}'.format(i, pivot_df.index[indices.flatten()[i]]))
    
    # Return the list of recommended anime names.
    return names_list

        

def unsupervised_user_based_recommender(movie_user_likes,n=200):
    '''
    The function unsupervised_user_based_recommender takes a user
    input anime title, and returns recommendations of similar animes
    using unsupervised user-based collaborative filtering. It loads
    necessary data, finds the closest anime title to the user input,
    and calls the reco function to get the recommended anime titles.
    '''
    #df = joblib.load(processed_data + "/" + "_anime_to_compare_with_name.pkl")    

    # Load the anime data with features to compare
    df = pd.read_csv(processed_data + "/" + "_anime_to_compare_with_name.csv") 
    
    # Convert the input anime title to lowercase
    lowertittle = movie_user_likes.lower() 
    
    #pivot_df_try = joblib.load(processed_data + "/" + "_to_find_index_user_based_unsupervised.pkl")

    # Load the pivot table to find the index of the input anime title
    shutil.unpack_archive(processed_data + "/" + "pivot_user_based_unsupervised.zip",processed_data)
    pivot_df_try = pd.read_csv(processed_data + "/" + "_to_find_index_user_based_unsupervised.csv")
    
    # Find the closest title to the input title based on string similarity
    closest_title, distance_score = finding_the_closest_title(lowertittle,df)
    
    # When the user input has no spelling mistakes
    if distance_score == 100:
        # Print the recommendations for similar animes to the closest title
        print('These are the recommendations for similar animes to '+'\033[1m'+str(closest_title)+'\033[0m'+'','\n')
        return reco(lowertittle,n,pivot_df_try)
    
    # When the user input has spelling mistakes
    else:
        # Print a message asking if the user meant the closest title found
        print('I guess you misspelled the name\n\nAre you looking similitudes for the anime named '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n' + '\nHere are the recommendations:\n')
        return reco(closest_title,n,pivot_df_try)



##############################################################
##############################################################
#                                                            #
#----------- Supervised User based recommendation -----------#
#                                                            #
##############################################################
##############################################################


def create_dict_su(final_df,gen,typ,method,n=100):
    '''
    This function takes a dataframe, genre, type, method, and n as input. 
    It filters the dataframe based on the specified genre and type using 
    either "or" or "and" method, selects the top n rows from the filtered 
    dataframe, and returns a dictionary representation of the resulting dataframe.
    '''
    # get the final dataframe and the parameters for filtering and number of recommendations to show
    df = final_df
    
    # check which method was used to filter the recommendations, 'or' or 'and'
    if method == 'or':
        # filter the dataframe using the OR logic and the given genres and types
        final_df = filtering_or(df, gen, typ)
    elif method == 'and':
        # filter the dataframe using the AND logic and the given genres and types
        final_df = filtering_and(df, gen, typ)
    else:
        # raise an error if an invalid filter type was given
        raise ValueError("Invalid filter type. Expected 'or' or 'and'.")
        
    # select the top n recommendations from the filtered dataframe
    final_df = final_df.head(n)
    
    # if the filtered dataframe is empty, print a message
    if final_df.empty:
        sentence = print('WOW!!!! Sorry, there is no matches for the anime and options selected! \n Try again, you might have more luck')
        return sentence
    else:
        # convert the filtered dataframe to a dictionary
        final_dict = final_df.to_dict('records')
        
        # return the dictionary of recommendations
        return final_dict





'''
The function takes an anime ID and uses a trained Singular Value Decomposition (SVD) 
algorithm to predict the estimated score of all other animes in the dataset. 
The function then sorts the predicted scores in descending order and returns
the resulting dataframe with anime names and their predicted scores.
'''
def sort_it(id):
    # Load the pre-trained SVD model
    algo = joblib.load(saved_models_folder + "/" + "SVD_new_model.pkl")
    
    # Load the anime dataframe
    df = pd.read_csv(processed_data + "/" + "anime_final.csv")
    
    # Apply the SVD model to estimate the score for each anime
    df['Estimate_Score'] = df['anime_id'].apply(lambda x: algo.predict(id, x).est)
    
    # Sort the dataframe by the estimated score in descending order and drop the anime_id column
    df = df.sort_values('Estimate_Score', ascending=False).drop(['anime_id'], axis = 1)
    
    # Create a blank index for the dataframe
    blankIndex=[''] * len(df)
    
    # Set the blank index to the dataframe
    df.index=blankIndex 
    
    # Return the sorted dataframe
    return df


