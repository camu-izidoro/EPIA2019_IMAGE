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
from sklearn.utils import shuffle
import numpy as np
import pandas as pd 
import pickle as pkl
from pathlib import Path
from sklearn import datasets, svm, metrics
data = []
labels = []
datatest=[]

df = pd.DataFrame(columns=['NomeImagem', 'Imagem'])
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
        df= df.append({'NomeImagem': numero_pasta, 'Imagem': original_image}, ignore_index=True)
df = shuffle(df)
export_csv = df.to_csv('export_dataframe.csv', index = None, header=None) #Don't forget to add '.csv' at the end of the path
print(df)
