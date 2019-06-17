#%%
import numpy as np
import cv2  as cv
import  matplotlib.pyplot as plt
import os

faceCascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
# Read image from your local file system
original_image = cv.imread('001_expresion_frown_3.bmp',0)

plt.figure(figsize=(20,10))
plt.imshow(original_image,cmap='gray')
plt.show()
# Detect faces
faces = faceCascade.detectMultiScale(
original_image,
scaleFactor=1.1,
minNeighbors=5,
flags=cv.CASCADE_SCALE_IMAGE
)
# For each face
for (x, y, w, h) in faces: 
    # Draw rectangle around the face
    sub_face = original_image[y:y+h, x:x+w]
    cv.imwrite("crop1.jpg",sub_face)
    cv.rectangle(original_image, (x, y), (x+w, y+h), (255, 255, 255), 3)

plt.figure(figsize=(20,10))
plt.imshow(original_image,cmap='gray')
plt.show()
#%%