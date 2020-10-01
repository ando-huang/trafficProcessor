# Traffic Light Processing Model

For: Brooklyness's Chasson Helmet, for scooter and bike riders.

Goal: Build Deep Learning model to recognize whether or not customers are obeying traffic lights

Input: Video from the front-facing camera

Output: Rating from 0 - 100 on how well the customer obeys traffic lights in a particular video

Trained on: 
https://hci.iwr.uni-heidelberg.de/node/6132/download/635332d553013d5afadcaef7db3b50a4

How it works (Conceptual):
  Takes snapshots at intervals of .5s throughout the video, and changes interval based on image class
    Classes are [Red, Yellow, Green, None] for images, all images fit into one of these four classes
  Determines class of the image using the model
  Uses class information in a series in order to determine a score for each instance of a traffic light
  Provides average of traffic scores as a result to each video given
