# Image Classification Flask App
Overview
This Flask application allows users to upload images and classify them using a pre-trained Support Vector Machine (SVM) model with Histogram of Oriented Gradients (HOG) features.

### Requirements
Python 3.x
OpenCV
scikit-learn
scikit-image
joblib
Flask
Setup
Clone this repository to your local machine.

Install the required Python packages using the following command:

Copy code
pip install -r requirements.txt
Ensure you have a directory structure similar to the one described below:

Copy code
dataset_full/
├── category_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── category_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
Usage
Activate your Python environment.
Run the command python app.py to start the Flask application.
Open the provided URL in your web browser to access the application.
Use the provided interface to upload an image file.
The uploaded image will be classified using the pre-trained SVM model, and the classification result will be displayed.
### Main Code Snippet
python
Copy code
import os
import sys
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

#Function to extract features from an image using Histogram of Oriented Gradients (HOG)
def extract_features(image, resize_shape=(128, 128)):
    #Implementation details omitted for brevity
    pass

#Function to classify an image using a pre-trained SVM model
def getImageClassification(image):
    # Implementation details omitted for brevity
    pass

#Function to fetch image files and their labels from a directory
def fetch_images(directory):
    # Implementation details omitted for brevity
    pass

#Function to create and train a SVM model
def modelCreation():
    # Implementation details omitted for brevity
    pass
## Authors
Rohan Amrutkar
### License
This project is licensed under the MIT License.
