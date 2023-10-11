import folium
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
import base64
from IPython.display import IFrame
from streamlit_folium import st_folium
from datetime import datetime

st.set_page_config(layout="wide")
path = (
    os.getcwd() + "/" + ".." + "/" + "data" + "/" + "train_data" + "/" + "metadata.csv"
)

metadata = pd.read_csv(path)
metadata["color"] = np.where(metadata["plume"] == "yes", "#00ff00", "#ff0000")
metadata["date"] = metadata["date"].apply(lambda x: datetime.strptime(str(x), "%Y%m%d"))
metadata["count"] = metadata.groupby(["lat", "lon"]).transform("size")
metadata["last_date"] = metadata.groupby(["lat", "lon"])["date"].transform("max")
metadata["plume_at_max_date"] = metadata[metadata["date"] == metadata["last_date"]][
    "plume"
]
metadata = metadata[metadata["plume_at_max_date"].isna() == False]

plume_dic = {"yes": "Leakage", "no": "No leakage"}
metadata = metadata.replace({"plume_at_max_date": plume_dic})

leak_bool = st.selectbox(
    "What do you wish to see?", np.array(["Leakages", "Non-leakages", "Both"])
)

if leak_bool == "Leakages":
    selected_metadata = metadata[metadata["plume"] == "yes"]
    selected_metadata = selected_metadata.reset_index()


if leak_bool == "Non-leakages":
    selected_metadata = metadata[metadata["plume"] != "yes"]
    selected_metadata = selected_metadata.reset_index()


if leak_bool == "Both":
    selected_metadata = metadata
    selected_metadata = selected_metadata.reset_index()

# Make an empty map
# Make an empty map

n = folium.Map()
coords = []
for i in range(0, len(selected_metadata)):
    coords += [(selected_metadata.iloc[i]["lat"], selected_metadata.iloc[i]["lon"])]
south_west_corner = min(coords)
north_east_corner = max(coords)
n.fit_bounds([south_west_corner, north_east_corner])

left_col_color = "#3e95b5"
right_col_color = "#f2f9ff"

# add marker one by one on the map
image_path = (
    "C:/Users/joaoh/OneDrive/Imagens/20230101_methane_mixing_ratio_id_4928.jpeg"
)
for i in range(0, len(selected_metadata)):
    html = (
        f"""
        <h1> Site report </h1>
        <center> <table style="height: 126px; width: 305px;">
        <tbody>
        <tr>
        <td style="width: 250px;background-color: """
        + left_col_color
        + """;"><span style="color: #ffffff;">Date of last picture </span></td>
        <td style="width: 250px;background-color: """
        + right_col_color
        + """;">{}</td>""".format(selected_metadata.iloc[i]["last_date"])
        + """
        </tr>
        <tr>
        <td style="background-color: """
        + left_col_color
        + """;"><span style="color: #ffffff;">Last known leakage state </span></td>
        <td style="width: 250px;background-color: """
        + right_col_color
        + """;">{}</td>""".format(selected_metadata.iloc[i]["plume_at_max_date"])
        + """
        </tr>
        <tr>
        <td style="background-color: """
        + left_col_color
        + """;"><span style="color: #ffffff;">Total number of pictures </span></td>
        <td style="width: 250px;background-color: """
        + right_col_color
        + """;">{}</td>""".format(selected_metadata.iloc[i]["count"])
        + """
        </tr>
        </tbody>
        </table></center>
        """
    )
    iframe = folium.IFrame(html=html, width=500, height=200)
    popup = folium.Popup(iframe, max_width=500)
    if selected_metadata.iloc[i]["plume"] == "yes":
        folium.Marker(
            location=[
                selected_metadata.iloc[i]["lat"],
                selected_metadata.iloc[i]["lon"],
            ],
            popup=popup,
            icon=folium.DivIcon(
                html=f"""
                <div><svg>
                    <circle cx="5" cy="5" r="5" fill="#5fd32c" />
                </svg></div>"""
            ),
        ).add_to(n)
    else:
        folium.Marker(
            location=[
                selected_metadata.iloc[i]["lat"],
                selected_metadata.iloc[i]["lon"],
            ],
            popup=popup,
            icon=folium.DivIcon(
                html=f"""
                <div><svg>
                    <circle cx="5" cy="5" r="5" fill="#e93020" />
                </svg></div>"""
            ),
        ).add_to(n)


st_data = st_folium(n, width=1200)
