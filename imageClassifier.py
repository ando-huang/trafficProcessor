import matplotlib.pyplot as plt
import numpy as np
import os
#import PIL for opening images
import tensorflow as tf
import pandas as pd #this needs to work, to read the csv and get tags for each image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

'''import pathlib
dataset_url = 'https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset/download'
data_dir = tf.keras.utils.get_file('traffic_photos.zip', origin=dataset_url, untar=True, extract = True)
data_dir = pathlib.Path(data_dir)
'''

#Uploads from Local, might have to reogranize and classify the data.
#data_dir should be a directory of the folders with each classification
data_dir = os.listdir("archive/daySequence1/daySequence1/frames/")
#use this to get the corresponding filenames and the "stop" "go" tags for each file
df = pd.read_csv("archive/Annotations/Annotations/daySequence1/frameAnnotationsBULB.csv", sep=";", usecols = ['Filename', 'Annotation tag']) 

batch_size = 32

#image params for the camera feed
img_height = 640 
img_width = 960

#image params for the training set
test_height = 960
test_width = 1280


#Fix these two training data sets, currently the function fails
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#all 4 of the annotation tags from the csv
class_names = df.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

num_classes = 2

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, val_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

