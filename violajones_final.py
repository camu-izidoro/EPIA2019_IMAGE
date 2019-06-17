
import numpy as np
import cv2  as cv
import  matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

faceCascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
# Read image from your local file system
os.mkdir('./basedados')
for i in range(1,51):
    
    # ele vai criar uma pasta e vai entrar em cada imagem e cortar
    if len(str(i)) == 1:
        numero_pasta="00" + str(i)
    if len(str(i)) == 2:
        numero_pasta="0" + str(i)
    
    os.mkdir('./basedados/'+numero_pasta)
    directory = Path('C:/Users/camil/Desktop/Face Database part1/'+numero_pasta+'')
    files = directory.glob('*.bmp')
    contador=1
    for f in files:
        original_image = cv.imread(str(f),0)
        nome_imagem = os.path.basename(f)
        # Detect faces
        faces = faceCascade.detectMultiScale(
        original_image,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv.CASCADE_SCALE_IMAGE
        )
        # For each face
        cont=1
        for (x, y, w, h) in faces: 
            # Draw rectangle around the face
            
            sub_face = original_image[y:y+h, x:x+w]
            sub_face = cv.resize(sub_face,(185,185))
            
            cv.imwrite('./basedados/'+numero_pasta+'/'+nome_imagem,sub_face)
            contador= contador+1
