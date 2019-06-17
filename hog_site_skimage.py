#%%
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from skimage import data, exposure
import cv2 
from skimage import io

dados =[]
labels =[]
img = cv2.imread('001_crop_3.bmp')
img = cv2.resize(img,(185,185))
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


img2 = cv2.imread('004_crop_3.bmp')
img2 = cv2.resize(img2,(185,185))
image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3), visualize=True)
teste_vetor = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3),block_norm='L2-Hys',feature_vector=True)
dados.append(teste_vetor)
labels.append('1')
teste_vetor2 = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3),block_norm='L2-Hys',feature_vector=True)
dados.append(teste_vetor2)
labels.append('2')


k = 3
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(dados, labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)




ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
#%%