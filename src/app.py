import streamlit as st
from streamlit_option_menu import option_menu
from body import testing
from body import un_based_rate
from body import un_based_feat
from body import sup_id
from body import about
from body import about_me

st.set_page_config(layout='wide')

# Defines a sidebar menu using the st.sidebar function from the Streamlit library

# This opens a sidebar in the Streamlit app.
with st.sidebar:
    choose = option_menu("Anime System Recommendator", ["About", "Based on ratings", "Based on Features", "Using user ID", "Testing","About the Creator"],
                         icons=['house', 'camera fill', 'kanban', 'book','person lines fill', 'book'],
                         menu_icon="app-indicator", default_index=0,
                         styles={"container": {"padding": "5!important", "background-color": "#fafafa"},
                                "icon": {"color": "orange", "font-size": "25px"}, 
                                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "#02ab21"},
                                }
                        )



# This creates a drop-down menu in the sidebar with six options: 
# "About", "Based on ratings", "Based on Features", "Using user ID", "Testing", and "About the Creator". 
# The icons argument provides icons for each option. menu_icon sets the icon for the sidebar. 
# default_index sets the default option in the drop-down menu. styles sets the styling for the drop-down menu.


# This uses the choose variable to determine which option was selected in the drop-down menu. 
# Depending on the option selected, it calls a specific function to display the corresponding 
# content in the main panel of the Streamlit app.

if choose == "About":
    about.it_is_about()
        
elif choose == "Based on ratings":
    un_based_rate.uns_bara()

elif choose == "Based on Features":
    un_based_feat.uns_feat()

elif choose == "Using user ID":
    sup_id.user_id()

elif choose == "Testing":
    testing.test_it()

elif choose == "About the Creator":
    about_me.about_me()
