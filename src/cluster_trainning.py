import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import csv


def convtofloat(s):
    try:
        return float(s)
    except ValueError:
        return 0.

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]

dataset = np.loadtxt("tracking_person.csv", delimiter=",",converters={0:convtofloat,1:convtofloat,2:convtofloat,3:convtofloat,4:convtofloat})
# split into input (X) and output (Y) variables
X = dataset[:,0:81]

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=12).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print set(labels)
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

print ("Clusters parameters")
with open('tracking_person_cluster.csv', 'a') as f:
    writer=csv.writer(f)
    for i in xrange(n_clusters_):
        list = X[labels==i]
        for element in list:
            new_array = np.append(element, i)
            writer.writerow(new_array)







# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
