import folium
import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import helper


st.set_page_config(layout="wide", page_title="Historical Data")

#Markdown code to add colors, spacing, margins etc
st.markdown(
    """
    <style>

        .main > div {
            padding-top: 1.8rem;
        }
""", unsafe_allow_html=True)
st.markdown(
        """

    <style>
    .st-emotion-cache-nziaof{
        background-color: #227250;
    }
    """,
        unsafe_allow_html=True,
    )

st.markdown(
        """

    <style>
    .st-emotion-cache-pkbazv{
        color: #ffffff;
    }
    """,
        unsafe_allow_html=True,
    )

st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(34, 114, 80, 1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: rgb(255, 255, 255);
   overflow-wrap: break-word;
}
 div[data-testid="stHorizontalBlock"] {
            margin-bottom:-50px;
 }

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: White;
   font-size: 28px;
}
"""
, unsafe_allow_html=True)

#Placing logo on sidebar
with st.sidebar:
  i=0
  while i <=13:
    st.write("")
    i +=1
  st.image("logo.jpg", width=300)

#Reading data
path = (
    os.getcwd() + "/" + ".." + "/" + "data" + "/" + "train_data" + "/" + "metadata.csv")
metadata = pd.read_csv(path)

##Creating 4 columns to display KPI's
col0, col1, col2, col3 = st.columns(4)


# Number of continents for which we have images
col0.metric("\# continents", 4)

#Number of pictures taken
col1.metric(
    "\# pictures",
    len(metadata),
)

#Number of Leakages
col2.metric(
    "\# Leakages",
    len(metadata[metadata["plume"] == "yes"]),
)

#Number of non Leakages
col3.metric(
    "\# non-leakages",
    len(metadata[metadata["plume"] == "no"]),
)


#Markdown code to do color, spacing and formating changes on KPI containers
st.markdown(
    """
    <style>

        .st-emotion-cache-16tkdie {
            gap: 0.8rem;
        }
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>

        .st-emotion-cache-xvtzic {
            margin-top: -10px;
        }
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>

        .st-emotion-cache-90gy28 {
            margin-top: -5px;
        }
""", unsafe_allow_html=True)


st.markdown('----')

st.markdown(
    """
    <style>
    [data-baseweb="select"] {
        margin-top: -45px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#Processing the data to display KPI's
metadata = helper.webapp_data_processing(metadata)

leak_bool = st.selectbox(
    "What do you wish to see?", np.array(["Leakages", "Non-leakages", "Both"])
)

if leak_bool == "Leakages":
    selected_metadata = metadata[metadata["plume_at_last_date"] == 1]
    selected_metadata = selected_metadata.reset_index()


if leak_bool == "Non-leakages":
    selected_metadata = metadata[metadata["plume_at_last_date"] != 1]
    selected_metadata = selected_metadata.reset_index()


if leak_bool == "Both":
    selected_metadata = metadata
    selected_metadata = selected_metadata.reset_index()


#creating map and bounding it
n = folium.Map()
for i in range(0, len(selected_metadata)):
    bounds = helper.mapsize(selected_metadata)
n.fit_bounds([bounds[0], bounds[1]])


#color of columns in popups
left_col_color = "#227250"
right_col_color = "#A9A9A9"
# add marker one by one on the map
for i in range(0,len(selected_metadata)):
    html=helper.table(left_col_color, right_col_color, selected_metadata, i)
    iframe = folium.IFrame(html=html, width=305, height=150)
    popup = folium.Popup(iframe, max_width=305)
    if selected_metadata.iloc[i]["plume_at_last_date"] == 0:
        folium.Marker(
            location=[selected_metadata.iloc[i]['lat'], selected_metadata.iloc[i]['lon']],
            popup=popup,
            icon=folium.DivIcon(html=f"""
                <div><svg>
                    <circle cx="5" cy="5" r="5" fill="#A4DD66" />
                </svg></div>""")
        ).add_to(n)
    else:
        folium.Marker(
            location=[selected_metadata.iloc[i]['lat'], selected_metadata.iloc[i]['lon']],
            popup=popup,
            icon=folium.DivIcon(html=f"""
                <div><svg>
                    <circle cx="5" cy="5" r="5" fill="#e93020" />
                </svg></div>""")
        ).add_to(n)

#rendering map
st_data = st_folium(n, width=1300, height = 500)
