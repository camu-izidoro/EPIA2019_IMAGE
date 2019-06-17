#%%
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from skimage import data, exposure
import cv2

dados_hog = []
labels = []
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = True

for x in range(1,2):
    if len(str(x)) == 1:
        numero_pasta="00" + str(x)
    if len(str(x)) == 2:
        numero_pasta="0" + str(x)
    
    for y in range(1,3):

        image = data.load('C:/Users/camil/Desktop/EPIA2019/basedados/'+str(numero_pasta)+'/'+str(numero_pasta)+'_crop_'+str(y)+'.bmp')
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
        descriptor = hog.compute(image)
        dados_hog.append(descriptor)
        labels.append(str(numero_pasta)+'_crop_'+str(y)+'.bmp')
print(dados_hog)
print(labels)
with open('your_file.txt', 'w') as f:
    for item in dados_hog:
        f.write("%s" % item)
#%%
