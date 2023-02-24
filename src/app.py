import streamlit as st
from streamlit_option_menu import option_menu
from body import un_based_rate
from body import un_based_feat
from body import sup_id
from body import about
from body import about_me
from body import what_is
from body import an_info
from body import explore
from body import eda
from body import get_recom

st.set_page_config(layout='wide')

# Defines a sidebar menu using the st.sidebar function from the Streamlit library

# This opens a sidebar in the Streamlit app.
with st.sidebar:
    choose = option_menu("Menu", ["About this project","What is anime?","Exploring data","Get recommendations","EDA", "Anime Info","About the Creator","Explore"],
                         icons=['house','easel',"clipboard-data", '123',"graph-up", 'tv','person','person'],
                         menu_icon="cast", default_index=0,
                         styles={"container": {"padding": "5!important", "background-color": "#fafafa"},
                                "icon": {"color": "orange", "font-size": "25px"}, 
                                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "#02ab21"},
                                },
                        )

# This creates a drop-down menu in the sidebar with six options: 
# "About", "What is anime?", "Exploring data", "Get recommendations", "EDA", "Anime Info", and "About the Creator". 
# The icons argument provides icons for each option. menu_icon sets the icon for the sidebar. 
# default_index sets the default option in the drop-down menu. styles sets the styling for the drop-down menu.


# This uses the choose variable to determine which option was selected in the drop-down menu. 
# Depending on the option selected, it calls a specific function to display the corresponding 
# content in the main panel of the Streamlit app.

if choose == "About this project":
    about.it_is_about()

elif choose == "What is anime?":
    what_is.what_is()

elif choose == "Exploring data":
    explore.explore_data()

elif choose == "Get recommendations":
    get_recom.get_the_recom()

elif choose == "EDA":
    eda.eda_info()

elif choose == "Anime Info":
    an_info.info()

elif choose == "About the Creator":
    about_me.about_me()

elif choose == "Explore":
    explore.explore_data()    
