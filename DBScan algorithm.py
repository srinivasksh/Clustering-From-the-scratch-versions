import numpy as np
import csv
import copy
import math
import matplotlib.pyplot as plt

%matplotlib inline

# Load data set
with open("dataset1.txt") as f:
    data = [list(np.float_(line)) for line in csv.reader(f, delimiter=" ")]
ip_size = len(data)
print("Number of records in input file: %d" % ip_size)

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
num_of_clusters = dbscan_algo(ipdata,0.3,4)

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
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
