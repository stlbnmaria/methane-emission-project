import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import os

st.set_page_config(layout="wide")
path = os.getcwd() + "/" +  ".." + "/" + "data" + "/" + "train_data" + "/" + "metadata.csv"

data = pd.read_csv(path)

leak_bool = st.selectbox("Do you wish to see leakages?", np.array(["Yes", "No"]))

if leak_bool == "Yes":
    selected_data = data[data["plume"] == "yes"]

if leak_bool == "No":
   selected_data = data[data["plume"] == "no"]

st.map(selected_data)




st.title("Test")

