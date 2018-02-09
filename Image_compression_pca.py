
import numpy as np
from sklearn.decomposition import RandomizedPCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
rcParams['figure.figsize'] = 11.7,8.27
from scipy.interpolate import interp1d

filename = 'panda.png'
print('reading file')
img = mpimg.imread(filename) 
print(img.shape)
# plt.axis('off') 
# plt.imshow(img)

print('reshaping image into 2 dimensions for PCA')
img_r = np.reshape(img, (img.shape[0],img.shape[1]*img.shape[2] )) 
print(img_r.shape)

number_of_components = 64
print('transofming image with'+str(number_of_components)+'number of components')
ipca = RandomizedPCA(number_of_components).fit(img_r)
img_c = ipca.transform(img_r)
print('new shape of image after transformation')
print(img_c.shape)
print('Randomized PCA with 64 components:')
print(np.sum(ipca.explained_variance_ratio_))

print('inversing the transformation back to image')
temp = ipca.inverse_transform(img_c) 
print('reshaping back to three dimension')
temp = np.reshape(temp, (img.shape[0],img.shape[1],img.shape[2])) 
print(temp.shape)

m = interp1d([temp.min(),temp.max()],[0,1])
print('rescaling image this may take some time....')
for i in range(temp.shape[0]):
    print(i)
    for j in range(temp.shape[1]):
        for k in range(temp.shape[2]):
            temp[i][j][k] = float(m(temp[i][j][k]))


fig = plt.figure()
plt.axis('off') 
plt.imshow(temp)



fig.savefig('panda_compressed.png')

