#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nudenet import NudeClassifier
import cv2
import os
import shutil

# number of divided frame for each video
number_of_divide = 100

# start number of second (skip intro for videos)
sec = 20

classifier = NudeClassifier()

# xml file for parameters AI faces
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# directory of mp4 videos
files = os.listdir('videos/')

def Check(file,sec):
    video_name = './videos/' + file
    create_file_name = file.replace(".mp4","")
    print("Checking ----> '"+create_file_name + "'")
    try:
        os.mkdir('results/' + create_file_name)
    except:
        print("Folder '"+ create_file_name +"' Already exists on 'results'.")
    try:
        os.mkdir('Temp/' + create_file_name)
    except:
        print("Folder '"+ create_file_name + "' Already exists on 'Temp'.")

    # create video capture object
    data = cv2.VideoCapture(video_name)

    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(data.get(cv2.CAP_PROP_FPS))

    # calculate dusration of the video
    seconds = int(frames / fps)
    #print("duration in seconds:", seconds)
    duration = int(seconds / number_of_divide)
    #print(duration)

    vidcap = cv2.VideoCapture(video_name)

    # function to check each frame in video and get pure results for faces and nudity
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:

            # save frame as JPG file
            cv2.imwrite("Temp/"+str(create_file_name)+"/frame "+str(sec)+" sec.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 1024])
            # get nudity result for frame
            info_dict = classifier.classify("Temp/"+str(create_file_name)+"/frame "+ str(sec) + " sec.jpg")
            # read frame to detect faces
            image = cv2.imread("Temp/"+str(create_file_name)+"/frame "+str(sec)+" sec.jpg")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(300, 300),
            )

            #print("Found {0} faces!".format(len(faces)))
            if len(faces) >= 1:
                if faces[0][0] >= 600:
                    # check the rate of image nudity
                    if info_dict["Temp/"+str(create_file_name)+"/frame "+ str(sec) + " sec.jpg"]['safe'] >= 0.9:
                        #print(info_dict)
                        # save the final image in new file
                        cv2.imwrite("results/"+str(create_file_name)+"/frame " + str(sec) + " sec.jpg", image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 1024])  # save frame as JPG file
                        #print(faces)
                        print("Image Found ----> "+"'/results/" + create_file_name + "/frame " + str(sec) + " sec.jpg'")

        return hasFrames

    frameRate = duration
    success = getFrame(sec)
    while sec < seconds:
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)


for file in files:
    Check(file,sec)
    create_file_name = file.replace(".mp4","")
    shutil.rmtree("Temp/"+create_file_name)
