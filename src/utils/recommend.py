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

#Preparing folder variables
#os.chdir(os.path.dirname(sys.path[0])) # This command makes the notebook the main path and can work in cascade.

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
data_folder = (PROJECT_ROOT + "/" + "data")

saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")
content_based_supervised_data = (data_folder + "/" + "processed" + "/" + "content_based_supervised")



shutil.unpack_archive(processed_data + "/" + "features_user_based_unsupervised.zip",processed_data)
shutil.unpack_archive(processed_data + "/" + "pivot_user_based_unsupervised.zip",processed_data)

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
Function to return the anime name that mtches de index number
'''
def from_index_to_title(index,df):
    anime = df
    return anime[anime.index == index]['name'].values[0]


'''
Function to return the matched index number of the anime name
'''
def from_title_to_index(title,df):
    anime = df
    return anime[anime["name"]==title].index.values[0]


'''
Function to find the closest title, It uses Levenshtein Distance to calculate the differences between sequences
'''
def match_the_score(a,b):
   return fuzz.ratio(a,b)


'''
Function to return the most similar title to the name a user typed
'''
def finding_the_closest_title(title,df):
    anime = df
    levenshtein_scores = list(enumerate(anime['name'].apply(match_the_score, b=title))) # Create a list wuth the matchin fuzz.ratio puntuation
    sorted_levenshtein_scores = sorted(levenshtein_scores, key=lambda x: x[1], reverse=True) # Sort from higher to lower the scores
    closest_title = from_index_to_title(sorted_levenshtein_scores[0][0],anime) # Getting the closest anime name/title
    distance_score = sorted_levenshtein_scores[0][1] # Getting the score
    return closest_title, distance_score

'''
Function to apply the filters fo the df recommended with animes
'''
def filtering(df,gen,typ):
    
    if (gen != "All") and (typ != "All"):
        filtered = df[df['genre'].str.contains(gen, regex=False, case=False, na=False)]
        filtered = filtered[filtered['type'].str.contains(typ, regex=False, case=False, na=False)]
        return filtered

    elif  (gen == "All") and (typ != "All"):
        filtered = df[df['type'].str.contains(typ, regex=False, case=False, na=False)]
        return filtered

    elif  (typ == "All") and (gen != "All"):
        filtered = df[df['genre'].str.contains(gen, regex=False, case=False, na=False)]
        return filtered

    elif  (typ == "All") and (gen == "All"):
        return df

'''
Create a df of the anime matches with the filters selected
'''
def create_df(names,gen,typ,n=100):
    #anime = joblib.load(processed_data + "/" + "_anime_to_compare_with_name.pkl")
    anime = pd.read_csv(processed_data + "/" + "_anime_to_compare_with_name.csv")# load anime df
    final_df = anime[anime['name'].isin(names)]
    final_df = final_df.drop(columns=["anime_id", "members"])
    blankIndex=[''] * len(final_df)
    final_df.index=blankIndex
    final_df = filtering(final_df,gen,typ)
    to_return = final_df.head(n)
    if final_df.empty:
        sentence = print('WOW!!!! Sorry, there is no matches for the anime and options selected! \n Try again, you might have mroe luck')
        return sentence
    else:
        return to_return


'''
Create dict of records with the filters selected - each row becomes a dictionary where key is column name and value is the data in the cell
'''
def create_dict(names,gen,typ,n=100):
    #anime = joblib.load(processed_data  + "/" +  "_anime_to_compare_with_name.pkl")
    anime = pd.read_csv(processed_data + "/" + "_anime_to_compare_with_name.csv")# load anime df
    final_df = anime[anime['name'].isin(names)]
    final_df = final_df.drop(columns=["anime_id", "members"])
    blankIndex=[''] * len(final_df)
    final_df.index=blankIndex
    final_df = filtering(final_df,gen,typ)
    final_df = final_df.head(n)
    if final_df.empty:
        sentence = print('WOW!!!! Sorry, there is no matches for the anime and options selected! \n Try again, you might have mroe luck')
        return sentence
    else:
        final_dict = final_df.to_dict('records')

        return final_dict

############################################################
############################################################
#                                                          #
#----- Unsupervised User content based recommendation -----#
#                                                          #
############################################################
############################################################

'''
A function that returns the names of the similar animes
for Unsupervised User content based recommendation system
'''
def print_similar_animes(query):
    ind = joblib.load(saved_models_folder + "/" + "model_based_content.pkl") # Load the trained model
    #anime = joblib.load(processed_data + "/" + "_anime_to_compare_with_name.pkl")
    anime = pd.read_csv(processed_data + "/" + "_anime_to_compare_with_name.csv")# load anime df
    closest_title, distance_score = finding_the_closest_title(query,anime) # find the closest title
       
    if distance_score == 100: # When a user does not make misspellings
        names = []
        errors = []
        print('These are the recommendations for similar animes to '+'\033[1m'+str(query)+'\033[0m'+'','\n')
        found_id = from_title_to_index(query,anime) # return the matched index number of the anime name
        array = ind[found_id][1:] # return the matched index number of the anime name that user did input
        indi = np.where(array==found_id) # return the position of the anime index that user did input (if it is in the list)
        array = np.delete(array, indi) # erase the anime index that matches the anime name that used did input
        #array = array[0:n] # print the number of anime recommendations that userd chosed
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
        found_id = from_title_to_index(closest_title,anime) # return the matched index number of the anime name that user did input
        array = ind[found_id][1:] # create and array with anime indexes to recoomend according to the anime 
        indi = np.where(array==found_id) # return the position of the anime index that user did input (if it is in the list)
        array = np.delete(array, indi) # erase the anime index that matches the anime name that user did input
        #array = array[0:n] # print the number of anime recommendations that userd chosed
        for id in array:
            try :
                names.append(anime[anime.index == id]['name'].values[0])
            except IndexError :
                errors.append(id)
        return names


#############################################################
#############################################################
#                                                           #
#--------- Unsupervised User  based recommendation ---------#
#                                                           #
#############################################################
#############################################################

'''
Return a list with recommendations for the anime 
'''
def reco(name,n,df):
    model_knn = joblib.load(saved_models_folder + "/" +"model_matrix_user_based_unsupervised.pkl")
    #shutil.unpack_archive(processed_data + "/" + "pivot_user_based_unsupervised.zip",processed_data)
    #pivot_df = pd.read_csv(processed_data + "/" + "pivot_user_based_unsupervised.zip")# load anime df
    pivot_df = joblib.load(processed_data + "/" +"pivot_user_based_unsupervised.pkl")
    indl = from_title_to_index(name,df)   
    distances, indices = model_knn.kneighbors(pivot_df.iloc[indl,:].values.reshape(1, -1), n_neighbors = n+1)
    names_list = []
    for i in range(1, n+1):
        if i == 0:
            print('WOW!!!! Sorry, there is no matches for the anime and options selected!\nTry again, you might have more luck')
        else:
            names_list.append(pivot_df.index[indices.flatten()[i]])
            #print('{0}: {1}'.format(i, pivot_df.index[indices.flatten()[i]]))
    
    return names_list

        

'''
A function that returns the names of the similar animes
for Unsupervised User content based recommendation system
'''
def unsupervised_user_based_recommender(movie_user_likes,n=200):
    #df = joblib.load(processed_data + "/" + "_anime_to_compare_with_name.pkl")
    df = pd.read_csv(processed_data + "/" + "_anime_to_compare_with_name.csv")# load anime df
    lowertittle = movie_user_likes.lower() # Pasamos el titulo a minúsculas
    #pivot_df_try = joblib.load(processed_data + "/" + "_to_find_index_user_based_unsupervised.pkl")
    pivot_df_try = pd.read_csv(processed_data + "/" + "_to_find_index_user_based_unsupervised.csv")# load anime df
    closest_title, distance_score = finding_the_closest_title(lowertittle,pivot_df_try)
    # When a user does not make misspellings
    if distance_score == 100:
        print('These are the recommendations for similar animes to '+'\033[1m'+str(closest_title)+'\033[0m'+'','\n')
        return reco(lowertittle,n,pivot_df_try)
    # When a user makes misspellings    
    else:
        print('I guess you misspelled the name\n\nAre you looking similitudes for the anime named '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n' + '\nHere are the recommendations:\n')
        return reco(closest_title,n,pivot_df_try)


##############################################################
##############################################################
#                                                            #
#----------- Supervised User based recommendation -----------#
#                                                            #
##############################################################
##############################################################

## Recomendacion by user Id, how many results and gender
'''
Create a df of the anime matches with the filters selected
'''
def df_recommendation(id,n,gen,typ):
    final_df = reco_by_user(id,n,gen,typ)
    to_return = final_df
    blankIndex=[''] * len(final_df)
    final_df.index=blankIndex
    if final_df.empty:
        sentence = print('WOW!!!! Sorry, there is no matches for the anime and options selected! \n Try again, you might have mroe luck')
        return sentence
    else:
        return to_return

def dict_recommendation(id,n,gen,typ):
    final_df = reco_by_user(id,n,gen,typ)
    to_return = final_df
    blankIndex=[''] * len(final_df)
    final_df.index=blankIndex
    if final_df.empty:
        sentence = print('WOW!!!! Sorry, there is no matches for the anime and options selected! \n Try again, you might have mroe luck')
        return sentence
    else:
        final_dict = final_df.to_dict('records')
        return final_dict

def sort_it(que_user,df,n):
    algo = joblib.load(saved_models_folder + "/" + "SVD_samople_fit.pkl")

    df['Estimate_Score'] = df['anime_id'].apply(lambda x: algo.predict(que_user, x).est)
    df = df.sort_values('Estimate_Score', ascending=False).drop(['anime_id'], axis = 1)
    blankIndex=[''] * len(df)
    df.index=blankIndex 
    return df.head(n)

def reco_by_user(id,n,gen,typ):
    anime = pd.read_csv(raw_data + "/" + "anime.csv")
    anime_map = anime.copy()
    anime_map.drop(['episodes','members','rating'],axis=1, inplace=True)
    chosen_user = anime_map.copy()

    if (gen != "All") and (typ != "All"):
        filtered = chosen_user[chosen_user['genre'].str.contains(gen, regex=False, case=False, na=False)]
        filtered = filtered[filtered['type'].str.contains(typ, regex=False, case=False, na=False)]
        return sort_it(id,filtered,n)

    elif  (gen == "All") and (typ != "All"):
        filtered = chosen_user[chosen_user['type'].str.contains(typ, regex=False, case=False, na=False)]
        return sort_it(id,filtered,n)

    elif  (typ == "All") and (gen != "All"):
        filtered = chosen_user[chosen_user['genre'].str.contains(gen, regex=False, case=False, na=False)]
        return sort_it(id,filtered,n)

    elif  (typ == "All") and (gen == "All"):
        return sort_it(id,chosen_user,n)


