import os
import shutil
import pandas as pd

df = pd.read_csv("archive/Annotations/Annotations/daySequence1/frameAnnotationsBULB.csv", sep=";", usecols = ['Filename', 'Annotation tag'])
os.chdir("archive/daySequence1/daySequence1/frames/")
os.mkdir("stop")
os.mkdir("go")
os.mkdir("warning")
#now based on the df file, sort all the images into the right folders
for i, j in df.iterrows():
    #i is the index of the iterrows, j is a series
    # now just sort the corresponding image from /frames/ to the right directory
    if j[1] == 'stop':
        #print('stop')
        print(j[0] + j[1])
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        shutil.move(imgPath, "stop/" + imgPath)
        os.remove(imgPath)
    elif j[1] == 'go':
        #print(j[0] + "go")
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        shutil.move(imgPath, "go/" + imgPath)
        os.remove(imgPath)
    else:
        #print('warning')
        imgPath = j[0].split("/")[1] #should get the actual filename within the frames folder
        shutil.move(imgPath, "go/" + imgPath)
        os.remove(imgPath)

