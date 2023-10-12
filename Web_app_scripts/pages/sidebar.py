import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium
import zipfile
import helper
#sys.path.append("../..")
#from inference import torch_inference

st.set_page_config(layout="wide", page_title="Live Data")

#Markdown code to add colors, spacing, margins etc
st.markdown(
    """
    <style>

        .st-emotion-cache-1aehpvj {
            font-size: 0px;
        }
        .main > div {
            margin-top: -60px;
        }
        .st-emotion-cache-nziaof{
        background-color: #227250;
        }
        .st-emotion-cache-pkbazv{
        color: #ffffff;
        }
        .st-emotion-cache-1uixxvy{
            margin-bottom: 40px;
        }
        .st-emotion-cache-vskyf7{
            margin-bottom: 40px;
        }
        .st-emotion-cache-1lx94gx{
            margin-bottom: 40px;
        }
""", unsafe_allow_html=True)

customized_button = st.markdown("""
    <style >
    .stDownloadButton, div.stButton {text-align:center}
    .stDownloadButton button, div.stButton > button:first-child {
        background-color:  #e5e7e9 ;
        color:#000000;
        padding-left: 20px;
        padding-right: 20px;
    }

    .stDownloadButton button:hover, div.stButton > button:hover {
        background-color:  #e5e7e9 ;
        color:#000000;
    }
        }
    </style>""", unsafe_allow_html=True)

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
            margin-bottom:-40px;
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

#Creating three columns to upload images, metadata and to download predictions
col0, col1, col2 = st.columns(3)
with col0:
    images = st.file_uploader(label="Upload images")
with col1:
    metadata = st.file_uploader(label="Upload Metadata")

results = []
if images is not None:
    with zipfile.ZipFile(images, 'r') as zip_ref:
        zip_ref.extractall("extracted_images")


results_path = os.getcwd() + "\\pages\\submission_test_file.csv"
results_df = pd.read_csv(results_path)
results_csv = results_df.to_csv(index=False)


with col2:
    col2.write("")
    col2.write("")
    st.download_button(
   "Download predictions",
   results_csv,)

if metadata is not None:
    #Merging predictions with the metadata
    results_df = results_df["label"]
    metadata = pd.read_csv(metadata)
    metadata = pd.merge(results_df, metadata, left_index=True, right_index=True)
    metadata["plume"] = np.where(metadata["label"] > 0.5, "yes", "no")

    #Creating 4 columns to display KPI's
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
        "\# leakages",
        len(metadata[metadata["plume"] == "yes"]),
    )

    #Number of non leakages
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
                margin-top: -8px;
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
    unsafe_allow_html=True,)


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
    if len(selected_metadata) > 0:
        bounds = helper.mapsize(selected_metadata)
        n.fit_bounds([bounds[0], bounds[1]])

    #color of columns in popups
    left_col_color = "#227250"
    right_col_color = "#A9A9A9"
    # add marker one by one on the map
    if len(selected_metadata) > 0:
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
                            <circle cx="5" cy="5" r="5" fill="#5fd32c" />
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





