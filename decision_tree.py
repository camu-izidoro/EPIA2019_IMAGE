#%%
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
    

    #print("[INFO] training classifier...")
    
    print("%s %s" % (train, test))
    X_train, X_test, y_train, y_test = features_x[train], features_x[test], features_y[train], features_y[test]
    
    acuracia=[]
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acuracia_total.append(accuracy_score(y_test, pred))
    k=k+1

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 6), acuracia_total, color='red', linestyle='dashed', marker='o',  
                 markerfacecolor='blue', markersize=10)
plt.title('Acuracia Geral')  
plt.xlabel('Partição')  
plt.ylabel('Acurácia')
plt.show()
'''
    joblib.dump(model, 'knn_model.pkl')
    knn = joblib.load('knn_model.pkl')
    digits = []
    #digits.append(cv2.imread('testeee.bmp',0))
    print(y_test)
    hogs = X_test
    # apply k-NN model created in previous
    predictions = list(map(lambda x: predict(x), hogs))
    print(hogs)
    print(predictions)
'''


def predict(df):
    predict = model.predict(df.reshape(1,-1))[0]
    predict_proba = model.predict_proba(df.reshape(1,-1))
    #print(predict_proba[0][predict])
    return predict,predict_proba[0][predict]


#%%
