'''
    dataSorter is intended to prepare the data in archive/ for imageClassifier.py
    the problem with the given data is that it isn't sorted
    this should place images in folders with corresponding classnames that would help the model
'''

import os
import shutil
import pandas as pd
print(pd.__version__)

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
    #print(j[0].split("/")[1])
    if j[1] == 'stop': #first case
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        #move(source, dest)
        shutil.copy(imgPath, "stop")
        #os.remove(imgPath)
    elif j[1] == 'go':
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        shutil.copy(imgPath, "go")
        #os.remove(imgPath)
    else:
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        shutil.copy(imgPath, "warning")
        #os.remove(imgPath)