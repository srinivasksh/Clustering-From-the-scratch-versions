import numpy as np
import csv
import copy
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

%matplotlib inline

# Load dataset into pandas dataframe
data = pd.read_csv("dataset2.txt",sep=" ",header=None)
print("Number of records in input file: %d " %len(data))

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

## K-Means clustering
print("Clustering data using K-Means Cluster algorithm")
kmeans = KMeans().fit(data)
y_pred = kmeans.predict(data)
print("Number of clusters from K-Means: %d" %len(set(y_pred)))
print(" ")

plt.subplot(221)
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=y_pred)
plt.title("K-Means Clustering")
plt.show()

## K-Means clustering
print("Clustering data using K-Means for 4 clusters")
kmeans = KMeans().fit(data)
y_pred = kmeans.predict(data)
print(" ")

plt.subplot(221)
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=y_pred)
plt.title("K-Means Clustering - 4 Clusters")
plt.show()

# Load data set into list for DBScan
with open("dataset2.txt") as f:
    data = [list(np.float_(line)) for line in csv.reader(f, delimiter=" ")]

## Prefix each element with its index number
ipdata = []
for idx,elem in enumerate(data):
	ipdata.append([idx,elem])

data_class = {}
	
def chk_class_val(ip_idx):
	if ip_idx[0] not in data_class:
		return 'Undefined'
	else:
		if data_class[ip_idx[0]] == 'Noise':
			return 'Noise'
		else:
			return 'Class'
			
def RangeQuery(data_db,ip_pnt,eps):
	data_set = copy.deepcopy(data_db)
	neighbours_tmp = []
	#print("ip_pnt:")
	#print(ip_pnt)
	#data_set.remove(ip_pnt)
	for pnt in data_set:
		pnt1 = np.array(copy.deepcopy(ip_pnt[1]))
		pnt2 = np.array(copy.deepcopy(pnt[1]))
		pt_dist = math.sqrt(np.power(pnt1-pnt2,2).sum())
		if pt_dist < eps:
			neighbours_tmp.append(pnt)
	return neighbours_tmp

## DBSCAN Algorithm
def dbscan_algo(data_db,eps,minPts):
	cluster_ix = 0
	for pnt_p in data_db:
		if (chk_class_val(pnt_p) == 'Undefined'):
			neighbour_set = RangeQuery(data_db, pnt_p, eps)
			if len(neighbour_set) < minPts:
				data_class[pnt_p[0]] = 'Noise'
			else:
				cluster_ix += 1
				print("Creating cluster : %d" %cluster_ix)
				data_class[pnt_p[0]] = cluster_ix
				for each_pnt in neighbour_set:
					data_class[each_pnt[0]] = cluster_ix
				while(neighbour_set):
					pnt_q = neighbour_set[0]
					neighbours = RangeQuery(data_db, pnt_q, eps)
					if len(neighbours) >= minPts:
						for pnt_r in neighbours:
							if chk_class_val(pnt_r) != 'Class':
								if chk_class_val(pnt_r) == 'Undefined':
									#print("pnt_r:")
									#print(pnt_r)
									neighbour_set.append(pnt_r)
								data_class[pnt_r[0]] = cluster_ix
					neighbour_set.remove(pnt_q)
	
	return cluster_ix
					
## Execute DB Scan algorithm
print("Clustering data using DBScan with eps: %d and MinPts : %d" %(5,10))
num_of_clusters = dbscan_algo(ipdata,5,10)
print("Number of clusters from DBScan: %d" %num_of_clusters)

## Extract index and cluster id from dictionary
points_list = []
cluster_list = []
for k,v in data_class.items():
	 points_list.append(k)
	 cluster_list.append(v)

## Plot the points for each cluster (varied by color)	 
for ix in list(range(num_of_clusters)):
	cluster_ix = ix+1
	indexes = [points_list[i] for i,x in enumerate(cluster_list) if x == cluster_ix]
	x_val = [ipdata[i][1][0] for i in indexes]
	y_val = [ipdata[i][1][1] for i in indexes]
	label_nm='Cluster'+str(cluster_ix)
	plt.scatter(x_val,y_val,label=label_nm)
	
## Plot for Noise
indexes = [points_list[i] for i,x in enumerate(cluster_list) if x == 'Noise']
x_val = [ipdata[i][1][0] for i in indexes]
y_val = [ipdata[i][1][1] for i in indexes]
label_nm='Noise Cluster'
plt.scatter(x_val,y_val,label=label_nm)