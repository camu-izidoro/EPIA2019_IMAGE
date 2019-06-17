#%%
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
dados_hog = []
labels = []

for x in range(1,5):
    if len(str(x)) == 1:
        numero_pasta="00" + str(x)
    if len(str(x)) == 2:
        numero_pasta="0" + str(x)
    
    for y in range(1,5):

        image = data.load('C:/Users/camil/Desktop/EPIA2019/basedados/'+str(numero_pasta)+'/'+str(numero_pasta)+'_crop_'+str(y)+'.bmp')

        hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
                    cells_per_block=(4, 4),transform_sqrt=True, block_norm="L1")

        dados_hog.append(hog_image)
with open('your_file.txt', 'w') as f:
    for item in dados_hog:
        f.write("%s\n" % item)
print(dados_hog)

#%%
