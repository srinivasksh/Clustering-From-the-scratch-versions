from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv("dataset1.txt",sep=" ",header=None)

## K-Means clustering Elbow method to determine Number of Clusters
print("Plotting cluster vs SSE (Elbow method)")
sse_list = []
k=range(1,10)
for k_in in k:
	kmeanModel = KMeans(n_clusters=k_in).fit(data)
	kmeanModel.fit(data)
	sse_list.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

# Plot the elbow
plt.plot(k, sse_list, 'bx-')
plt.xlabel('k')
plt.ylabel('sse')
plt.title('Number of clusters vs SSE')
plt.show()

## K-Means
print("K-Means clustering with no parameters passed to function")
y_pred = KMeans().fit_predict(data)
print("Number of clusters selected by KMeans: %d" %len(set(y_pred)))
plt.subplot(221)
plt.scatter(data[0], data[1], c=y_pred,cmap=plt.cm.autumn)
plt.title("K-Means with no parameters")
plt.show()
	
## K-Means with 2 clusters
print("K-Means clustering with Number of clusters as 2")
y_pred = KMeans(n_clusters=2).fit_predict(data)
plt.subplot(221)
plt.scatter(data[0], data[1], c=y_pred,cmap=plt.cm.autumn)
plt.title("K-Means for 2 clusters")
plt.show()