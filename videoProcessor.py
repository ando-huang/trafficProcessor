# create a folder to store extracted images
import os

#create a folder with a name referencing the video
# use opencv to do the job
import cv2


print(cv2.__version__)  # my version is 3.1.0

def readVideo(path):

    vidcap = cv2.VideoCapture(path)
    folder = (os.path.splitext(path)[0]) + "IMAGEFILES"  
    os.mkdir(folder)
    count = 0.0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(folder,"frame{:.1f}.jpg".format(count)), image)     # save frame as JPEG file
        #count is the increment between each screencap
        count += .5
    count *=2 #there are count*2 images in the folder
    print("{} images are extacted in {}.".format(count,folder))

readVideo('2020_06_15_1_27_46.mp4')
