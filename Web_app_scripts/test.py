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

st.set_page_config(layout="wide", page_title="Historical Data")
path = (
    os.getcwd() + "/" + ".." + "/" + "data" + "/" + "train_data" + "/" + "metadata.csv"
)

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

with st.sidebar:
  i=0
  while i <=13:
    st.write("")
    i +=1
  st.image("logo.jpg", width=300)

metadata = pd.read_csv(path)
col0, col1, col2, col3 = st.columns(4)


#border: 1px solid rgba(34, 114, 80, 0.6);
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

# number of unhealthy+ (unhealthy, very unhealthy, hazardous)
# days for the Year - comparison prev year
col3.metric(
    "\# non-leakages",
    len(metadata[metadata["plume"] == "no"]),
)
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
metadata = pd.read_csv(path)

def webapp_data_processing(data):
    data['date'] =  data['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    data['count'] = data.groupby(['lat', "lon"]).transform('size')
    data['last_date'] = data.groupby(["lat", "lon"])['date'].transform('max')
    data["plume"] = np.where(data["plume"] == "yes",1, 0)
    data['plume_at_last_date'] = np.where(data['date'] == data['last_date'], np.where(data["plume"] == 1,1,0), 0)
    data["plume_at_last_date"] = data.groupby(["lat", "lon"])['plume_at_last_date'].transform('max')
    data['threshold_date'] = (data["date"] >= (data['last_date'] - pd.DateOffset(months=1)))
    data["plume_last_month"] =  np.where(((data['threshold_date']) & (data["plume"] == 1) ), True, False)
    data["last_month"] =  np.where((data['threshold_date']), True, False)
    data = (data.groupby(['lat','lon','count',"last_date", "plume_at_last_date"])
         .apply(lambda x: ((x['plume_last_month']).sum(), (x["last_month"]).sum(), (x["plume"]).sum()))
         .reset_index(name='New_count'))
    data['plume_count_lm'] = data["New_count"].apply(lambda x: x[0])
    data["total_count_lm"] = data["New_count"].apply(lambda x: x[1])
    data["plume_count"] = data["New_count"].apply(lambda x: x[2])
    data["Responsible"] = "joaohfpmelo@gmail.com"
    data = data.drop("New_count", axis=1)
    return data

metadata = webapp_data_processing(metadata)

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

# Make an empty map
# Make an empty map

n = folium.Map()
coords = []
for i in range(0, len(selected_metadata)):
    coords += [(selected_metadata.iloc[i]["lat"], selected_metadata.iloc[i]["lon"])]
south_west_corner = min(coords)
north_east_corner = max(coords)
n.fit_bounds([south_west_corner, north_east_corner])

left_col_color = "#227250"
right_col_color = "#A9A9A9"


# add marker one by one on the map
for i in range(0,len(selected_metadata)):
    html=f"""
        <center> <table style="height: 126px; width: 305px;">
        <tbody>
        <tr>
        <td style="width: 250px;background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Date of last picture </span></td>
        <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["last_date"]) + """
        </tr>
        <tr>
        <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Last known leakage state </span></td>
        <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["plume_at_last_date"]) + """
        </tr>
        <tr>
        <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Total number of pictures </span></td>
        <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["count"]) + """
        </tr>
        <tr>
        <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Total number of leakages </span></td>
        <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["plume_count"]) + """
        </tr>
        <tr>
        <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Number of pictures last month </span></td>
        <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["total_count_lm"]) + """
        </tr>
        <tr>
        <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Number of leakages last month </span></td>
        <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["plume_count_lm"]) + """
        </tr>
        <tr>
        <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Responsible </span></td>
        <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i][""]) + """
        </tr>
        </tbody>
        </table></center>
            """
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

st_data = st_folium(n, width=1300, height = 500)
