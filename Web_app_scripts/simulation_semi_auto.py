import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import os

image_path = (os.getcwd() + "/" + ".." + "/" + "data" + "/" + "train_data" + "/" + "plume" + "/")
meta_path = (os.getcwd() + "/" + ".." + "/" + "data" + "/" + "train_data" + "/" + "metadata.csv")

metadata_df = pd.read_csv(meta_path)


if 'current_index' not in st.session_state:    # Initialize or update session state for current index
    st.session_state.current_index = 0
else:
    st.session_state.current_index += 1
st.session_state.current_index %= len(metadata_df)

if st.button('Load New Image and Predict'): 
    current_metadata = metadata_df.iloc[st.session_state.current_index]
    img_path = os.path.join(BASE_PATH, current_metadata['path'])
    image = Image.open(img_path)

    st.image(image, caption='Satellite Image', use_column_width=True)
    st.write(current_metadata)

    prediction = model.predict(image) # Load model and predict 