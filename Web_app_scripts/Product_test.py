import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# import seaborn as sns
import os

st.set_page_config(layout="wide")
path = (
    os.getcwd() + "/" + ".." + "/" + "data" + "/" + "train_data" + "/" + "metadata.csv"
)

metadata = pd.read_csv(path)
metadata["color"] = np.where(metadata["plume"] == "yes", "#00ff00", "#ff0000")

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



fig = px.scatter_geo(
    data_frame=selected_metadata,
    lat="lat",
    lon="lon",
    color="color",
    color_discrete_map={"#00ff00": "green", "#ff0000": "red"},
)
fig.update_geos(fitbounds="locations")
fig.update_layout(showlegend=False)
fig.update_geos(
    resolution=50,
    showcoastlines=True,
    coastlinecolor="Gray",
    showland=True,
    landcolor="beige",
    showocean=True,
    oceancolor="LightBlue",
    showcountries=True,
    countrycolor="Gray",
)
fig.update_layout(height=600, width=750, margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)




st.title("Test")
