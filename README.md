# Traffic Light Processing Model

Goal: Build Neural Network model to recognize whether people are obeying traffic lights

Input: Video from the front-facing camera

Output: Rating from 0 - 100 on how well the customer obeys traffic lights in a particular video

Trained on: 
dataset from kaggle, **will link later**

How it works (Conceptual):
  Takes snapshots at intervals of 1s throughout the video, and changes interval based on image class
    Classes are [Red, Yellow, Green, None] for images, all images fit into one of these four classes
  Determines class of the image using the model
  Uses class information in a series in order to determine a score for each instance of a traffic light
  Provides average of traffic scores as a result to each video given

non-ML files:
dataSorter.py - reorganizes the file structure of the training data to be more like a sample t-set
videoProcessor.py - takes snapshots of the video we want to check, and then stores them in a folder

# Set up Environment if using venv

'''
1. activate venv 
  a. python3 -m venv env
  b. source env/bin/activate
  c. pip install {tensorflow, opencv-python}

2. deactivate