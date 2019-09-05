import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# x = [130, 167, 213, 441, 445, 451, 478, 515, 526, 564, 655, 782, 1261]
# x = [1,1,5,6,1,5,10,22,23,23,50,51,51,52,100,112,130,500,512,600,12000,12230]
x = [1,2,3,60,70,80,100,220,230,250]
kmeans = KMeans(n_clusters=2)
a = kmeans.fit(np.reshape(x,(len(x),1)))
centroids = kmeans.cluster_centers_

labels = kmeans.labels_
print(labels)