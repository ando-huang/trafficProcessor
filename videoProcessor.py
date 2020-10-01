import os
import cv2

'''
    readVideo CREATES a FOLDER and stores JPEGs at 1 second intervals
    Parameters:
        path - the filename of the video to process
'''
def readVideo(path):
    vidcap = cv2.VideoCapture(path)
    folder = (os.path.splitext(path)[0]) + "IMAGEFILES"  
    os.mkdir(folder)
    count = 0.0
    #while True:
    while True:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(folder,"frame{:.1f}.jpg".format(count)), image)     # save frame as JPEG file
        #count is the increment between each screencap
        count += 1
    count *=2 #there are count*2 images in the folder
    print("{} images are extacted in {}.".format(count,folder))

#driver code, runs through all the files available in working directory
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    breaks = f.split('.')
    if breaks[1] == "MP4":
        readVideo(f)
