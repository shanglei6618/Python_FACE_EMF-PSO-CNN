Python 3.7.7 (tags/v3.7.7:d7c567b08f, Mar 10 2020, 10:41:24) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # -*- coding: utf-8 -*-
# import the necessary packages
#####  Face feature point extraction

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np #
import argparse
import imutils
import time
import dlib
import cv2
 
 
def eye_aspect_ratio(eye):
    # （X，Y）coordinates
    A = dist.euclidean(eye[1], eye[5])# Calculate the Euclidean distance between two sets
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    # return EAR
    return ear
 
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    # EAR calculation
    mar = (A + B) / (2.0 * C)
    return mar
 
    
    # EAR calculation

def face_aspect_ratio(eye):
    # （X，Y）coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # FAR calculation
    far = (A + B) / (2.0 * C)
    # return FAR
    return far 
    
 
# Initialize the DLIB's Face detector (HOG) and then create face marker predictions
print("[INFO] loading facial landmark predictor...")
# 1：Use dlib.get_frontal_face_detector() to get a face position detector
detector = dlib.get_frontal_face_detector()
# 2：Use dlib.shape_predictor to obtain a facial feature location detector
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')# The file needs to be in the same path as the code
 
# 3：Get an index of the left and right eye facial markers
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 4：Turn on the cv2 local camera
# cap = cv2.VideoCapture(0)

# If you cannot open the camera, you can use the following function to read the video file
cap = cv2.VideoCapture('file name.mp4')# The video file needs to be in the same path as the code
 
# Define an empty list
a = []
b = []
c = []   
# Loop frames from the video stream
while True:
    # 5：Carry on the loop, read the picture, and enlarge the dimension of the picture, and grayscale
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 6：Face position detection was performed using detector(gray, 0)
    rects = detector(gray, 0)
    
    # 7：Use predictor(gray, rect) to cycle for the location of facial features
    for rect in rects:
        shape = predictor(gray, rect)
        
        # 8：Converts facial feature information into array format
        shape = face_utils.shape_to_np(shape)
        
        # 9：
        #Extract left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # Mouth coordinate
        mouth = shape[mStart:mEnd]
        
        #1023-Extract face contour coordinates
        new_face = shape[[1,6,10,15,24,19]]
        
        
        # 10：
        # EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # MAR
        mar = mouth_aspect_ratio(mouth)
        
        #FAR: Use the same formula as the EAR
        face = face_aspect_ratio(new_face)
 
        # 11：Using cv2.convexHull, draw the outline position using drawContours for drawing operations
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
 
        # 12：Carry out drawing operation, use rectangular box to mark the face
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)    
        
        
        
        # 13：Displays the values of EAR,MAR, and FAR
 

 
        cv2.putText(frame, "   EAR: {:.2f}".format(ear), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        


        cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
           
       
        cv2.putText(frame, "FAR: {:.2f}".format(face), (480, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        
        # 14：Drawing operation, 68 feature points identification
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    
        print('MAR:{:.2f} '.format(mar),
              'EAR:{:.2f} '.format(ear),
             'FAR:{:.2f} '.format(face))
        # Face feature points are stored in the list
        a.append(ear.copy())

        b.append(mar.copy())

        c.append(face.copy())

    

        
    # Press q to exit
    cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # show with opencv
    cv2.imshow("Frame", frame)
    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
