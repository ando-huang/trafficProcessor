import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import zipfile
import urllib, urllister

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Import the training dataset from (BOSCH)
data_url = 'https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1681105773/dataset_test_riib.zip.001'
#god i hate underscores


#now do something with the data_path to extract the images

batch_size = 32
#these image size params match the extracted images from sebastian's video
img_height = 640
img_width = 960

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2
    subset = "training"
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2
    subset = "validation"
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size)

#takes the classifications from the dataset
class_names = train_ds.class_names
#print(class_names)
#we want to have ['red', 'yellow', 'green', 'off']
#FOUR CLASSSIFICATIONS


