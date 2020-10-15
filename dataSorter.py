'''
    *** TARGETS DAYSEQUENCE1 ***
    dataSorter is intended to prepare the data in archive/ for imageClassifier.py
    the problem with the given data is that it isn't sorted
    this should place images in folders with corresponding classnames that would help the model
'''

import os
import shutil
import pandas as pd

#read in the csv corresponding to this particular folder of data
df = pd.read_csv(
    "archive/Annotations/Annotations/daySequence1/frameAnnotationsBULB.csv",
    sep=";",
    usecols = ['Filename', 'Annotation tag']
    )
os.chdir("archive/daySequence1/daySequence1/frames/")
#print(os.listdir())
files = os.listdir()
#only create the new folders if they dont exist already
if 'stop' not in files:
    os.mkdir("stop")
if 'go' not in files:
    os.mkdir("go")
if 'warning' not in files:
    os.mkdir("warning")

# i is index, j is series of data about the file at that index
for i, j in df.iterrows():
    # sort each image into the right folder based on the csv
    if j[1] == 'stop': #first case
        imgPath = j[0].split("/")[1] #gets the filename inside frames folder
        shutil.copy(imgPath, "stop")
        
    elif j[1] == 'go':
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        shutil.copy(imgPath, "go")
        
    else:
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        shutil.copy(imgPath, "warning")

print("Sorting complete")