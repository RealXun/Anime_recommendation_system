import streamlit as st
import os
import sys
from utils import recommend
from PIL import Image
import pickle
import requests
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO


def uns_bara():

    #Add the cover image for the cover page. Used a little trick to center the image
            # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Unsupervised user based collaborative filtering</p>', unsafe_allow_html=True)



 # The code prompts the user to enter the name of an anime using a text input field.
 # It checks if the input is valid by verifying that it is not an empty string and does 
 # not contain only numbers. If the input is valid, the code displays a message to 
 # confirm that the input is valid. If the input is not valid, the code displays an 
 # error message asking the user to enter a string with at least one non-numeric character.
    
    to_search = st.text_input("Please write the name of the anime:")   # a text input field for user to input name of the anime to search for
    if to_search:   # checks if user has entered any value
        if to_search.isnumeric():   # checks if the value entered by the user is a number
            st.write("Input contains only numbers. Please enter a string with at least one non-numeric character.")   # displays error message if the input contains only numbers
        else:
            st.write(f"Input is valid: {to_search}")   # displays success message if the input is valid



# This code prompts the user to enter the maximum number of recommendations 
# they would like to receive, then it attempts to convert the user input into 
# an integer using a try-except block. If the user input can be converted to 
# an integer, it is stored in the variable number_of_recommendations. If the 
# conversion fails, an error message is displayed. If the user input is an 
# integer, a success message is displayed.

    # Prompts the user to enter a maximum number of recommendations 
    # they want to get and stores the input in the 'user_input' variable.
    user_input  = st.text_input("What is the maximum number of recommendations you would like to get?:") 

    # The input is then converted into an integer type and stored in the 
    # 'number_of_recommendations' variable using the 'int' function. 
    try:
        number_of_recommendations = int(user_input)
    except ValueError:
        st.error("Please enter a valid integer.")

    # If the input is successfully converted to an integer, a success message 
    # is displayed with the integer value using the 'st.success' function.
    if isinstance(user_input, int):
        st.success(f"You entered the integer: {user_input}")

  

# The code presents a dropdown menu to select between two filtering methods ("and" and "or"). 
# Depending on the method chosen, the user can select one or more genres and one or more types 
# of anime using checkboxes. If the "and" method is chosen, the user can only select one type 
# of anime, while the "or" method allows the user to select multiple types. The selected genres 
# and types are stored in the selected_genre and selected_type variables.

    method = st.selectbox("Choose a filtering method", ["and", "or"]) # prompts user to choose filtering method either 'and' or 'or'

    if method == "or": # if filtering method is 'or'

        option_genre = ['ALL','Drama', 'Romance', 'School', 'Supernatural', 'Action', # list of anime genres
        'Adventure', 'Fantasy', 'Magic', 'Military', 'Shounen', 'Comedy',
        'Historical', 'Parody', 'Samurai', 'Sci-Fi', 'Thriller', 'Sports',
        'Super Power', 'Space', 'Slice of Life', 'Mecha', 'Music',
        'Mystery', 'Seinen', 'Martial Arts', 'Vampire', 'Shoujo', 'Horror',
        'Police', 'Psychological', 'Demons', 'Ecchi', 'Josei',
        'Shounen Ai', 'Game', 'Dementia', 'Harem', 'Cars', 'Kids',
        'Shoujo Ai', 'Hentai', 'Yaoi', 'Yuri']
        option_type = ['ALL','Movie', 'TV', 'OVA', 'Special', 'Music', 'ONA'] # list of anime types

        selected_genre = st.multiselect('Select genre', option_genre) # prompts user to select genres
        selected_type = st.multiselect('Select type', option_type) # prompts user to select anime types

    else: # if filtering method is 'and'
        st.text("AND method would match any gender you input with the type.\n More Genres, more results \n Type should be one, there is no anime with two types at once")

        option_genre = ['Drama', 'Romance', 'School', 'Supernatural', 'Action', # list of anime genres
        'Adventure', 'Fantasy', 'Magic', 'Military', 'Shounen', 'Comedy',
        'Historical', 'Parody', 'Samurai', 'Sci-Fi', 'Thriller', 'Sports',
        'Super Power', 'Space', 'Slice of Life', 'Mecha', 'Music',
        'Mystery', 'Seinen', 'Martial Arts', 'Vampire', 'Shoujo', 'Horror',
        'Police', 'Psychological', 'Demons', 'Ecchi', 'Josei',
        'Shounen Ai', 'Game', 'Dementia', 'Harem', 'Cars', 'Kids',
        'Shoujo Ai', 'Hentai', 'Yaoi', 'Yuri']
        option_type = ['Movie', 'TV', 'OVA', 'Special', 'Music', 'ONA'] # list of anime types

        selected_genre = st.multiselect('Select genre', option_genre) # prompts user to select genres
        selected_type = st.multiselect('Select type', option_type, max_selections=1) # prompts user to select anime types, allowing only one selection



    def unsupervised_user_explicit_rating_based(name,n,genre,type,method):
        similar_animes = recommend.create_dict(recommend.unsupervised_user_based_recommender(name,n),genre,type,method)
        return similar_animes



    criteria_selected = to_search and user_input and selected_genre and selected_type



    # Create a visual indicator to show if both criteria are selected
    if criteria_selected:
        st.success('All criteria are selected. You can click now.')
    else:
        st.warning('Please select All criteria to get recommendations')




# Displays anime recommendations based on selected criteria. It uses the Streamlit library 
# to create a user interface with input fields for selecting anime genre, type, and method 
# for recommendation. When the user clicks the "Get the Recommendation" button, the script 
# retrieves anime recommendations based on the selected criteria using a pre-trained model. 
# It then displays the recommendations in a grid of images and text information such as 
# English and Japanese titles, type, episodes, duration, rating, and score. If there are 
# no recommendations to display or the user has not entered enough information, the script 
# prompts the user accordingly.
 


    # Enable button if both criteria are selected
    if st.button('Get the Recommendation', disabled=not criteria_selected):
        with st.spinner('Generating recommendations...'):
            result = unsupervised_user_explicit_rating_based(to_search,number_of_recommendations,selected_genre,selected_type,method)
            if result is not None: 
                # If the recommendation results are not empty, create a new dictionary to store them
                new_dict={}
                for di in result:
                    # For each recommendation, add a new entry to the dictionary using the anime name as the key
                    new_dict[di['name']]={}
                    # Copy all other properties of the recommendation into the dictionary entry for that anime
                    for k in di.keys():
                        if k =='name': continue
                        new_dict[di['name']][k]=di[k]

                # Determine how many rows and columns are needed to display the recommendations
                num_cols = 5
                num_rows = len(result) // num_cols + 1

                # Loop through each row of recommendations
                for row_idx in range(num_rows):
                    # Create a new set of columns to display each recommendation
                    cols = st.columns(num_cols)
                    # Loop through each column and get the key (anime name) for that column's recommendation
                    for col_idx, key in enumerate(list(new_dict.keys())[row_idx*num_cols:(row_idx+1)*num_cols]):
                        # Get the recommendation for the current anime
                        result = new_dict[key]

                        # Get the cover image for the anime from the recommendation data
                        response = requests.get(result['cover'])
                        img = Image.open(BytesIO(response.content))

                        # Display the anime information in a container within the current column
                        with cols[col_idx].container():
                            cols[col_idx].image(img, use_column_width=True)
                            cols[col_idx].write(f"**{result['english_title']}**")
                            if 'japanese_title' in result:
                                cols[col_idx].write(f"**{result['japanese_title']}")
                            if 'type' in result:
                                cols[col_idx].write(f"**Type:** {result['type']}")
                            if 'episodes' in result:
                                cols[col_idx].write(f"**Episodes:** {result['episodes']}")
                            if 'duration' in result:
                                cols[col_idx].write(f"**Duration:** {result['duration']}")
                            if 'rating' in result:
                                cols[col_idx].write(f"**Rating:** {result['rating']}")
                            if 'score' in result:
                                cols[col_idx].write(f"**Score:** {result['score']}/10")
                            # Display the estimated score for the recommendation
                            if 'Estimate_Score' in result:
                                    cols[col_idx].write(f"**{float(result['Estimate_Score'])}**")

            else:
                # If there are no recommendations to display, inform the user
                st.write("Sorry, there is no matches for this, try again with different filters.")
                
            # If the user has not entered enough information to get recommendations, prompt them to do so
    else :
        st.write("")