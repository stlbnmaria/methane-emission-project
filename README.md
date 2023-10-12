# Methane Emission Project

Authors: Alvaro Calafell, João Melo, Steve Moses, Harshit Shangari, Thomas Schneider & Maria Stoelben

## Description

## Data
The data consists of satellite images of different locations. There are 428 annotated images and 108 test images. The data is labeled with whether a location contains a methane plume or not. The image size is 64 x 64 pixels with one channel, i.e. grayscale images. Additionally, metadata of the images was provided, incl. longitude, latitude, date and coordinates of the plume. Below example images from the data as well as the geographical locations are displayed.

<p float="left">
  <img src='EDA/example_img.png' width="46%" />
  <img src='EDA/map.png' width="53%" /> 
</p>

## Setup
Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To install requirements:

```bash
pip install -r requirements.in
```

It is assumed that the user has the data and stored it in the root directory of the repo as indicated below:

    .
    ├── ...
    └── data                  
        ├── test_data
        │   ├── images    
        │   │   └── ...      
        │   └── metadata.csv          
        └── train_data
            ├── images    
            │   ├── no_plume   
            │   │   └── ... 
            │   └── plume  
            │       └── ...     
            └── metadata.csv 


## Run Training
```bash
python train.py
```

## Run Inference
```bash
python inference.py
```

## Results
