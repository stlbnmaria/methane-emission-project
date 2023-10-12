# Methane Emission Project

Authors: Alvaro Calafell, João Melo, Steve Moses, Harshit Shangari, Thomas Schneider & Maria Stoelben

## Description

## Data
The data consists of satellite images of different locations. There are 428 annotated images and 108 test images. The data is labeled with whether a location contains a methane plume or not. The image size is 64 x 64 pixels with one channel, i.e. grayscale images. Additionally, metadata of the images was provided, incl. longitude, latitude, date and coordinates of the plume. Below example images from the data as well as the geographical locations are displayed.

<p float="left">
  <img src='EDA/example_img.png' width="46%" />
  <img src='EDA/map.png' width="53%" /> 
</p>

## Data Augmentation
We use image augmentation techniques to augment the training data since the size of our dataset is small and we run the risk of overfitting. We use different geometric transformations such as random cropping, rotations, horizontal and vertical flips as well as adjust the sharpness and contrast of the original images to create new augmented images for the training data. Finally, we normalize all our images. For the validation data, the images are only resized and cropped depending on the input requirement of the model we use and normalized in the end.

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

## Run the App

## Results
Explain CV -split used for results 
batch size, folds, epochs

5 - fold cross validation
final model trained on all data
batch size: 32 after augmentation
10 epochs
save model with highest Val AUC

Model | Avg. Val AUC | Weigths
--- | --- | ---
Baseline CNN | 0.86 | None
ResNet18 | 0.96 | IMAGENET1K_V1
DenseNet-121 | 0.95 | IMAGENET1K_V1
Swin-T | 0.94 | IMAGENET1K_V1
VGG19-BN | TBD | IMAGENET1K_V1
ResNet50 | 0.91 | IMAGENET1K_V2