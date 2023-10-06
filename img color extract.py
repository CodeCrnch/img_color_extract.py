import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from PIL import Image


Image.open("sample.jpg")
image = mpimg.imread("sample.jpg")
w, h, d = tuple(image.shape)
pixels = np.reshape(image, (w*h, d))

n_colors = 10
model = KMeans(n_clusters=n_colors, n_init=10, random_state=42).fit(pixels)
palette = np.uint8(model.cluster_centers_)
plt.imshow([palette])
plt.show()



#pip install scikit-learn
