import streamlit as st
import os
import sys
from utils import recommend
from PIL import Image
import pickle
import requests
from pathlib import Path
from streamlit_option_menu import option_menu



with st.sidebar:
    choose = option_menu("Anime System Recommendator", ["About", "Based on ratings", "Based on Features", "Using user ID", "Other", "Based on Features2"],
                         icons=['house', 'camera fill', 'kanban', 'book','person lines fill', 'book'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "About":
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About the Creator</p>', unsafe_allow_html=True)
    st.write("Aquí pondría mi introducción o lo que sea")    

    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About the work</p>', unsafe_allow_html=True)
    st.write("Explicar el trabajo")  
        
elif choose == "Based on ratings":
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.title('Anime Recommendation System')
    st.title('Unsupervised user based collaborative filtering')


    # Get the user's favorite movie
    to_search = st.text_input('What anime would you like to search for recommendations?')

    # Get the user's favorite movie
    number_of_recommendations = st.slider('How many recommendations would you like to get?', min_value=1, max_value=100, value=5, step=1)
    st.write('The current number is ', number_of_recommendations)

    def ratings_based(name,genre,type,n):
        similar_animes = recommend.create_df(recommend.unsupervised_user_based_recommender(name),genre,type,n)
        return similar_animes

    ## Drop down menu to select the genre
    option_gere = st.selectbox('What kind of genre would you like to search (you can choose all)',('All','Email', 'Home phone', 'Mobile phone'))
    st.write('You selected:', option_gere)

    ## Drop down menu to select the type
    option_type = st.selectbox('What type of anime would you like to search (you can choose all)',('All','TV', 'OVA', 'ONA'))
    st.write('You selected:', option_type)
    if (st.button('Get the Recommendation')):
        # dataframe = load('../models/df.pkl')
        result = ratings_based(to_search,option_gere,option_type,number_of_recommendations)
        st.dataframe(result)
        st.balloons()

elif choose == "Based on Features":
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.title('Anime Recommendation System')
    st.title('Unsupervised user based collaborative filtering')


    # Get the user's favorite movie
    to_search = st.text_input('What anime would you like to search for recommendations?')

    # Get the user's favorite movie
    number_of_recommendations = st.slider('How many recommendations would you like to get?', min_value=1, max_value=100, value=5, step=1)
    st.write('The current number is ', number_of_recommendations)

    def features_based(name,genre,type,n):
        similar_animes = recommend.create_df(recommend.print_similar_animes(name),genre,type,n)
        return similar_animes

    ## Drop down menu to select the genre
    option_gere = st.selectbox('What kind of genre would you like to search (you can choose all)',('All','Email', 'Home phone', 'Mobile phone'))
    st.write('You selected:', option_gere)

    ## Drop down menu to select the type
    option_type = st.selectbox('What type of anime would you like to search (you can choose all)',('All','TV', 'OVA', 'ONA'))
    st.write('You selected:', option_type)
    if (st.button('Get the Recommendation')):
        # dataframe = load('../models/df.pkl')
        result = features_based(to_search,option_gere,option_type,number_of_recommendations)
        st.dataframe(result)
        st.balloons()

elif choose == "Using user ID":
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.title('Anime Recommendation System')
    st.title('Supervised user rating based collaborative filtering')


     # Get the user's favorite movie
    users_id = st.slider('How many recommendations would you like to get?', min_value=1, max_value=25000, value=5, step=1)
    st.write('The current number is ', users_id)

    # Get the user's favorite movie
    number_of_recommendations = st.slider('How many recommendations would you like to get?', min_value=1, max_value=100, value=5, step=1)
    st.write('The current number is ', number_of_recommendations)

    def super_ratings_based(id,n,genre,type):
        similar_animes =recommend.df_recommendation(id,n,genre,type)
        return similar_animes


    ## Drop down menu to select the genre
    option_gere = st.selectbox('What kind of genre would you like to search (you can choose all)',('All','Email', 'Home phone', 'Mobile phone'))
    st.write('You selected:', option_gere)

    ## Drop down menu to select the type
    option_type = st.selectbox('What type of anime would you like to search (you can choose all)',('All','TV', 'OVA', 'ONA'))
    st.write('You selected:', option_type)
    if (st.button('Get the Recommendation')):
        # dataframe = load('../models/df.pkl')
        result = super_ratings_based(users_id,number_of_recommendations,option_gere,option_type)
        st.dataframe(result)
        st.balloons()

elif choose == "Other":
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    import streamlit as st

    # Example data
    data = [{"name": "The Shawshank Redemption", "genre": "Drama", "type": "Movie"},        {"name": "The Godfather", "genre": "Crime", "type": "Movie"},        {"name": "The Godfather: Part II", "genre": "Crime", "type": "Movie"},        {"name": "The Dark Knight", "genre": "Action", "type": "Movie"},        {"name": "12 Angry Men", "genre": "Drama", "type": "Movie"},        {"name": "Schindler's List", "genre": "Biography", "type": "Movie"},        {"name": "The Lord of the Rings: The Return of the King", "genre": "Adventure", "type": "Movie"},        {"name": "Pulp Fiction", "genre": "Crime", "type": "Movie"},        {"name": "The Good, the Bad and the Ugly", "genre": "Western", "type": "Movie"}]

    # Number of results
    n = len(data)

    # Columns
    col1 = data[:n//3]
    col2 = data[n//3:2*n//3]
    col3 = data[2*n//3:]

    # Display the columns
    st.write("Column 1")
    for item in col1:
        st.write("-", item["name"], item["genre"], item["type"])

    st.write("Column 2")
    for item in col2:
        st.write("-", item["name"], item["genre"], item["type"])

    st.write("Column 3")
    for item in col3:
        st.write("-", item["name"], item["genre"], item["type"])



elif choose == "Based on Features2":
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.title('Anime Recommendation System')
    st.title('Unsupervised user based collaborative filtering')


    # Get the user's favorite movie
    to_search = st.text_input('What anime would you like to search for recommendations?')

    # Get the user's favorite movie
    number_of_recommendations = st.slider('How many recommendations would you like to get?', min_value=1, max_value=100, value=5, step=1)
    st.write('The current number is ', number_of_recommendations)

    def fetch_poster(movie_id):
        response = requests.get('https://cconnect.s3.amazonaws.com/wp-content/uploads/2020/03/Funko-Pop-Hunter-x-Hunter-Figures-thumb-600.jpg')
        data = response.json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

    def features_based(name,genre,type,n):
        similar_animes = recommend.create_dict(recommend.print_similar_animes(name),genre,type,n)
        recommended_movies = []
        recommended_movies_posters = []
        for x in similar_animes:
            #movie_id = movies.iloc[x[0]].movie_id
            recommended_movies.append(x["name"])
            #recommended_movies_posters.append(fetch_poster(movie_id))
        return recommended_movies


    ## Drop down menu to select the genre
    option_gere = st.selectbox('What kind of genre would you like to search (you can choose all)',('All','Email', 'Home phone', 'Mobile phone'))
    st.write('You selected:', option_gere)

    ## Drop down menu to select the type
    option_type = st.selectbox('What type of anime would you like to search (you can choose all)',('All','TV', 'OVA', 'ONA'))
    st.write('You selected:', option_type)
    if (st.button('Get the Recommendation')):
        # dataframe = load('../models/df.pkl')
        st.title("\n")
        # recommendations show
        st.subheader("Recommendations for you")
        st.subheader("\n")

        my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        n = 3 #elements per sublist
        
        final_list = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )]


        names = features_based(to_search,option_gere,option_type,number_of_recommendations)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(names[0])
            #st.markdown("![Alt Text](https://cconnect.s3.amazonaws.com/wp-content/uploads/2020/03/Funko-Pop-Hunter-x-Hunter-Figures-thumb-600.jpg)")
        with col2:
            st.text(names[1])
            #st.markdown("![Alt Text](https://cconnect.s3.amazonaws.com/wp-content/uploads/2020/03/Funko-Pop-Hunter-x-Hunter-Figures-thumb-600.jpg)")
        with col3:
            st.text(names[2])
            #st.markdown("![Alt Text](https://cconnect.s3.amazonaws.com/wp-content/uploads/2020/03/Funko-Pop-Hunter-x-Hunter-Figures-thumb-600.jpg)")
        with col4:
            st.text(names[3])
            #st.markdown("![Alt Text](https://cconnect.s3.amazonaws.com/wp-content/uploads/2020/03/Funko-Pop-Hunter-x-Hunter-Figures-thumb-600.jpg)")
        with col5:
            st.text(names[4])
            #st.markdown("![Alt Text](https://cconnect.s3.amazonaws.com/wp-content/uploads/2020/03/Funko-Pop-Hunter-x-Hunter-Figures-thumb-600.jpg)")
