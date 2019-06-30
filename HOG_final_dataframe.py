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
import csv
data = []
labels = []
datatest=[]

# loop over the image paths in the training set
with open('export_dataframe.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        my_data = np.array(row[1],dtype=int) 
        print(my_data)
	    # extract Histogram of Oriented Gradients from the logo
        H = feature.hog(my_data, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4))
	    # update the data and labels
     
        data.append(H)
        labels.append(str(row[0]))

with open("train.pkl", "wb") as f:
    pkl.dump([data, labels], f)

#%%
