import streamlit as st
import os
import sys
from scripts import head_intro
from scripts import main_body
from scripts import body_based_on_ratings
from scripts import body_based_on_features
from scripts import body_by_user_id
from scripts import *
from utils import page_config
from PIL import Image


def page_config():
    st.set_page_config(
        page_tittle = 'Anime Recommendation Systems',
        page_icon = 'images/cover.png',
        layout = 'wide'
    )
# Slide bar
    slidebar = st.slidebar.selectbox(
        'Menu',
        ('Intro','Based on ratings')
        )
    if slidebar =='Intro': # Based in ratings
        # Title and description
        head_intro()
        # Body content  
        main_body()
    else: # Based in ratings
        # Title and description
        #head_one()
        # Body content  
        body_based_on_ratings()

    #if slidebar =='Based on features': # Based on features
    #    # Title and description
    #    #head_one()
    #    # Body content
    #    body_based_on_features()  

    #else : # By User ID
    #    # Title and description
    #    #head_one()
    #    # Body content
    #    body_by_user_id()