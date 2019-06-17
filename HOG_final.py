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
for x in range(1,20):
    if len(str(x)) == 1:
        numero_pasta="00" + str(x)
    if len(str(x)) == 2:
        numero_pasta="0" + str(x)
    
    directory = Path('C:/Users/camil/Desktop/EPIA2019/basedados/'+numero_pasta)
    contador=1
    files = directory.glob('*.bmp')

    
   
    for f in files:
        original_image = cv2.imread(str(f),0)
	    # extract Histogram of Oriented Gradients from the logo
        H = feature.hog(original_image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(8, 8))
	    # update the data and labels
     
        data.append(H)
        labels.append(str(numero_pasta))
        contador=contador+1
        arquivo.write(str(numero_pasta) +'_'+ str(contador)+'\n')
        print(str(numero_pasta) +'_'+ str(contador))
arquivo.close()

with open("train.pkl", "wb") as f:
    pkl.dump([data, labels], f)

'''
header1 = labels
np.savetxt('imagensposhog.dat', data, header=header1)
'''

'''
features  = np.array(data, 'float64')
tamanho=(len(data)*80)/100
tamanhoteste=len(data) - tamanho
arrendondado=round(tamanhoteste)


X_train, X_test, y_train, y_test = train_test_split(features, labels)

# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)
model_score = model.score(X_test, y_test)
joblib.dump(model, 'knn_model.pkl')

knn = joblib.load('knn_model.pkl')


def feature_extraction(image):
    return feature.hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(8, 8))
def predict(df):
    predict = knn.predict(df.reshape(1,-1))[0]
    predict_proba = knn.predict_proba(df.reshape(1,-1))
    #print(predict_proba[0][predict])
    return predict
digits = []

#digits.append(cv2.imread('testeee.bmp',0))
print(y_test)
hogs = X_test
# apply k-NN model created in previous
predictions = list(map(lambda x: predict(x), hogs))
print(hogs)
print(predictions)

print("[INFO] evaluating...")
contador=round(tamanhoteste)
arquivo2 = open('predicao.txt','w')
for a in data[round(tamanhoteste): len(data)]:
    print('imagem buscada' +labels[contador] )
    pred = model.predict(a.reshape(1, -1))[0]
    print('imagem que encontrou' + pred)
    contador = contador + 1 
    arquivo2.write('imagembuscada' + labels[contador] + '_imagemencontrada'+ pred+'\n')
arquivo2.close()
'''
#%%
