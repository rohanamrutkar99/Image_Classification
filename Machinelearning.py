import os
import sys
import cv2
import numpy as np
from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from skimage.feature import hog
import joblib
directory = "D:\Spark\spark_Notebook\dataset_1\dataset_full"
## to extract features from Image.
def extract_features(image,resize_shape=(128, 128)):
    """
    This function extracts Histogram of Oriented Gradients (HOG) features from a resized grayscale
    image.
    
    :param image: The `image` parameter is the input image that you want to extract features from. This
    function takes an image as input and processes it to extract Histogram of Oriented Gradients (HOG)
    features. The image can be in color or grayscale format
    :param resize_shape: The `resize_shape` parameter is a tuple that specifies the dimensions to which
    the image should be resized. In this case, the default value is set to (128, 128), meaning the image
    will be resized to a shape of 128x128 pixels. You can adjust this parameter to resize
    :return: The function `extract_features` returns the Histogram of Oriented Gradients (HOG) features
    extracted from the input image after resizing and converting it to grayscale.
    """
    try:
        # Convert the image to grayscale
        # Resize the image to a fixed size
        resized_image = cv2.resize(image, resize_shape)        
        # Convert the resized image to grayscale
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # Extract Histogram of Oriented Gradients (HOG) features
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

        return features
    except Exception as e:
        print(e)
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))

 


def getImageClassification(image):
    """
    The function `getImageClassification` extracts features from an image, loads a pre-trained SVM model
    or creates a new one, and then predicts the classification of the image.
    
    :param image: The code snippet you provided seems to be a function for image classification using a
    SVM model. It first extracts features from the input image, then checks if the SVM model file
    exists. If the model file exists, it loads the model and makes a prediction on the image features.
    If the model file does
    :return: the classification result of the input image after processing it through a machine learning
    model.
    """
    try:
        features = extract_features(image)
        if  os.path.exists('svm_model.pkl'):
            loadModel = joblib.load('svm_model.pkl')
            classificationResult =loadModel.predict([features])
        else:
            modelCreation()
            loadModel = joblib.load('svm_model.pkl')
            classificationResult = loadModel.predict([features])
        print(2)
        return classificationResult
    except Exception as e:
        print(e)
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))


def fetch_images(directory):
    """
    The function `fetch_images` retrieves image files from a specified directory along with their
    corresponding labels.
    
    :param directory: The `directory` parameter in the `fetch_images` function is the path to the
    directory where the images are stored. This function will search through this directory and its
    subdirectories to find image files (based on common image file extensions like .png, .jpg, .jpeg,
    .gif) and
    :return: The `fetch_images` function returns two lists: `image_files` containing the full paths of
    image files found in the specified directory, and `labels` containing the labels extracted from the
    directory names where the images are located.
    """
    image_files = []
    labels = []
    # Iterate over all files and directories in the given directory
    for root, dirs, files in os.walk(directory):
        # Check if the current directory contains image files
        for file in files:
            # Check if the file is an image (you can extend this condition based on your image file extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Construct the full path of the image file and append it to the list
                image_files.append(os.path.join(root, file))
                labels.append(root.split('\\')[-1])
    return image_files,labels


def modelCreation():
    """
    The function `modelCreation` fetches images, extracts features, splits the dataset, trains a Support
    Vector Machine classifier, saves the model, makes predictions, and calculates accuracy.
    """
    try:
        image_paths,labels = fetch_images(directory)
        features = []
        for image_path in image_paths:

            image = cv2.imread(image_path)

            features.append(extract_features(image))
        
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # Train Support Vector Machine (SVM) classifier
        clf = svm.SVC(kernel='poly')

        clf.fit(X_train, y_train)
        ###to make model pickled
        # Make predictions on test set
        joblib.dump(clf, 'svm_model.pkl')
        y_pred = clf.predict(X_test)       
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)
    except Exception as e:
        print(e)
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
