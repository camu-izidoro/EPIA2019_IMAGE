#%%
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import exposure
from skimage import io
from sklearn.neighbors import KNeighborsClassifier


def monta_hog(n_amostras, dataset):
    lista_hog = []
    for i in range(0, n_amostras):
        nome_img = './basedados/' + dataset + '/'+ dataset +'_crop_' + str(i+1) + '.bmp'

        image = io.imread(nome_img, as_grey = True)
        hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
                    cells_per_block=(4, 4),transform_sqrt=True, block_norm="L1")

        # reescala histograma para melhor visualizacao
        #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range = (0, 0.02))
        lista_hog.append(hog_image)

    # exibe a ultima imagem
    #plt.subplot(1, 2, 1) # imagem em um grid 1 x 2, 1a posicao do grid
    #plt.title("Imagem original")
    #plt.imshow(image, cmap='gray')

    ##plt.subplot(1, 2, 2) # imagem em um grid 1 x 2, 2a posicao do grid
    #plt.title("HOG")
    #plt.imshow(hog_image, cmap='gray')

    plt.show()

    lista_hog = np.array(lista_hog)
    lista_hog = lista_hog.reshape((n_amostras, -1)) # ajusta dimensoes da lista para ficar compativel com o classificador
    return lista_hog


lista_hog_001 = monta_hog(5, '001')
lista_hog_002= monta_hog(5, '002')
# print lista_hog_cow
# print lista_hog_dog

# organiza amostras equilibradamente e monta o vetor de classes
amostras = []
classes = []
for i in range(0, 5):
    amostras.append(lista_hog_001[i])
    classes.append('001')
    amostras.append(lista_hog_002[i])
    classes.append('002')

# separacao treinamento e teste
proporcao_treinamento = 0.8
caracteristicas_treinamento = amostras[:int(proporcao_treinamento * len(amostras))]
caracteristicas_teste = amostras[int(proporcao_treinamento * len(amostras)):]
classes_treinamento = classes[:int(proporcao_treinamento * len(amostras))]
classes_teste = classes[int(proporcao_treinamento * len(amostras)):]

# print caracteristicas_treinamento
# print caracteristicas_teste
# print classes_treinamento
# print classes_teste

# treinamento
k = 3
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(caracteristicas_treinamento, classes_treinamento)
classes_previstas = neigh.predict(caracteristicas_teste)

# verifica acertos
porcentagem_acerto = 0
for i in range(0, len(classes_teste)):
    print(classes_teste[i] + ' - ' + classes_previstas[i])
    if classes_teste[i] == classes_previstas[i]:
        porcentagem_acerto += 1
        print (porcentagem_acerto)
porcentagem_acerto /= float(len(classes_teste))
print(porcentagem_acerto * 100)

#%%
