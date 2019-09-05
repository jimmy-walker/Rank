import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# #############################################################################
# Generate sample data
# X = [1,2,4,7,9,5,4,7,9,56,57,54,60,200,297,275,243]
X = [130, 167, 213, 441, 445, 451, 478, 515, 526, 564, 655, 782, 1261]
X = np.reshape(X, (-1, 1))

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)

ms = MeanShift(bandwidth=None, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
print(labels)