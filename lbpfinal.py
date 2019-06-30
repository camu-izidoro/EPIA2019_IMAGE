#%%
# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from tempfile import TemporaryFile
from skimage import exposure,color
from skimage import feature
import argparse
import imutils
import cv2
import os
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn import datasets, svm, metrics
data = []
labels = []
datatest=[]
arquivo = open('labelsimagens.txt','w')
# loop over the image paths in the training set
for x in range(1,54):
    if len(str(x)) == 1:
        numero_pasta="00" + str(x)
    if len(str(x)) == 2:
        numero_pasta="0" + str(x)
    
    directory = Path('./basedados/'+numero_pasta)
    contador=1
    files = directory.glob('*.bmp')

    
   
    for f in files:
        original_image = cv2.imread(str(f),0)
        lbp = feature.local_binary_pattern(original_image, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
 
		# normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        data.append(hist)
        labels.append(str(numero_pasta))
        contador=contador+1
        arquivo.write(str(numero_pasta) +'_'+ str(contador)+'\n')
        print(f.stem)
arquivo.close()

with open("train_lbp.pkl", "wb") as f:
    pkl.dump([data, labels], f)

#%%
