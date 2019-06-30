#%%
import math
import operator
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from tempfile import TemporaryFile
from skimage import exposure,color
from sklearn.metrics import accuracy_score
from skimage import feature
import argparse
import imutils
import cv2
import os
from pathlib import Path
from sklearn import datasets, svm, metrics

def DistanciaEuclidiana(tupla1,tupla2,tamanho):
    distancia=0
    for x in range(tamanho):
        distancia += pow((tupla1[x] - tupla2[x]),2)
    return math.sqrt(distancia)


def getNeighbors(data_train, labels_train,testInstance,testlabel, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(data_train)):
		dist = DistanciaEuclidiana(testInstance, data_train[x], length)
		distances.append((labels_train[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors



def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(label_test, predictions):
	correct = 0
	for x in range(len(label_test)):
		if label_test[x] == predictions[x]:
			correct += 1
	return (correct/float(len(label_test))) * 100.0
	

with open("train.pkl", "rb") as f:
        X, Y = pkl.load(f)

#print(X)
#print(Y)
kf = KFold(n_splits=5,shuffle=True)
k=1
features_x  = np.array(X, 'float64')
features_y  = np.array(Y, 'str')
acuracia_total=[]
for train, test in kf.split(features_x):	
    print("%s %s" % (train, test))
    X_train, X_test, y_train, y_test = features_x[train], features_x[test], features_y[train], features_y[test]
    predictions=[]
    k = 3
    for x in range(len(X_test)):
        neighbors = getNeighbors(X_train,y_train, X_test[x],y_test[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(y_test[x]))
    accuracy = getAccuracy(y_test, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')  
    acuracia_total.append(accuracy) 

print('Total Accuracy: ' + repr(acuracia_total))
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 6), acuracia_total, color='red', linestyle='dashed', marker='o',  
                 markerfacecolor='blue', markersize=10)
plt.title('Acuracia Geral')  
plt.xlabel('Partição')  
plt.ylabel('Acurácia')
plt.show()

#%%