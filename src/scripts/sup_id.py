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
import pandas as pd
import xlsxwriter
import base64

output = BytesIO()

def to_excel(df):
    # Create a BytesIO object to store the Excel file as bytes
    output = BytesIO()
    
    # Create a Pandas ExcelWriter object with the XlsxWriter engine
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Write the DataFrame to the Excel file, specifying the sheet name and that the index should not be included
    df.to_excel(writer, index=False, sheet_name='Recommendations')
    
    # Get a reference to the XlsxWriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Recommendations']
    
    # Create a format for numbers with two decimal places
    format1 = workbook.add_format({'num_format': '0.00'}) 
    
    # Apply the number format to the first column of the worksheet
    worksheet.set_column('A:A', None, format1)  
    
    # Save the Excel file and get its contents as bytes
    writer.save()
    processed_data = output.getvalue()
    
    # Return the Excel file contents as bytes
    return processed_data



def user_id():
#Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Supervised Collaborative Filtering based on ratings</p>', unsafe_allow_html=True)


# Getting user input in the form of an integer for the ID of a user for whom 
# the anime recommendation will be generated. The code first asks the user 
# to input the ID, then it tries to convert the user input to an integer. 
# If it's successful, it returns a success message with the integer value. 
# If it's not successful, it returns an error message asking the user to 
# enter a valid integer.

    user_input_1  = st.text_input("Choose the ID of the user you would like to see recommendations") # create a text input for the user to enter the ID of the user they want recommendations for
    try:
        users_id = int(user_input_1) # convert the input to an integer
    except ValueError:
        st.error("Please enter a valid integer.") # show an error message if the input is not a valid integer

    if isinstance(user_input_1, int): # check if the input is an integer
        st.success(f"You entered the integer: {user_input_1}") # show a success message with the input value



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

        option_genre = ['Drama', 'Romance', 'School', 'Supernatural', 'Action', # list of anime genres
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



    def super_ratings_based(id,n,genre,type, method):
        if method == "and":
            similar_animes = recommend.create_dict_su(recommend.sort_it(id), genre, type,method,n)
        else:
            similar_animes = recommend.create_dict_su(recommend.sort_it(id),genre,type,method,n)
        
        return similar_animes



    criteria_selected = user_input_1 and user_input and selected_genre and selected_type



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
            result = super_ratings_based(users_id,number_of_recommendations,selected_genre,selected_type, method)
            if result is not None: 

                # Define a dataframe from the result list
                df = pd.DataFrame(result)

                # Call the function to create a excel file
                df_xlsx = to_excel(df)

                # Button to download the excel file
                st.download_button(label='???? Download Recommendations',
                                                data=df_xlsx ,
                                                file_name= 'Recommendations.xlsx')


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
                                    cols[col_idx].write(f"Estimated Score: {result['Estimate_Score']:.2f}")

            else:
                # If there are no recommendations to display, inform the user
                st.write("Sorry, there is no matches for this, try again with different filters.")
                
            # If the user has not entered enough information to get recommendations, prompt them to do so
            if not (users_id and number_of_recommendations):
                st.write("")
    else :
        st.write("")

