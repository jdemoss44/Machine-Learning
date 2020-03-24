import numpy as np
import matplotlib.pyplot as plt
import collections as Counter
import pandas as pd



heightweight = pd.read_csv("weight-height.csv")
# print(heightweight.iloc[0:10,0])
# print(len(heightweight.Gender))

def data_split (percentage, data):
	pos = round(len(data.iloc[:,0])/100)*round(percentage)
	X_train = data.iloc[0:pos,1:]
	y_train = data.iloc[0:pos,0]
	X_test = data.iloc[pos:,1:]
	y_test = data.iloc[pos:,0]
	return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = data_split(80, heightweight)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# wh_norm_hist = np.histogram(heightweight[heightweight.Gender=="Male"]["Weight"], bins= 10,density=False)
# # print(wh_norm_hist)
# wy = wh_norm_hist[0]
# wx = wh_norm_hist[1][:-1]
# # print(wy) #this is the y values (number of ppl)
# # print(wx) #this is the x val (weight/height)
# # plt.plot(wx, wy)
# # plt.show()

from sklearn.datasets import load_iris
iris_dataset = load_iris()
# print(iris_dataset)
# print("Keys of iris_dataset:\n", iris_dataset.keys())
# print("Shape of data:", iris_dataset['data'].shape)
# print("First five rows of data:\n", iris_dataset['data'][:5])
# print("Type of target:", type(iris_dataset['target']))
# print("Shape of target:", iris_dataset['target'].shape)

# split the data up into training and testing:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)

# Fit the model to the data
# from sklearn.neighbors import KNeighborsClassifier
# kn = KNeighborsClassifier(n_neighbors=1)
# kn.fit(X_train, y_train)

def L2 (X_train, X_test, pos):
	distance = np.sqrt(np.sum(np.square(X_test-X_train.iloc[pos,:])));
	return distance

def L1 (X_train, X_test, pos):
	distance = np.sum(np.abs(X_test-X_train.iloc[pos,:]));
	return distance



def predict(X_train, y_train, k, metric, X_test):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute a distance
		distance = metric(X_train, X_test, i)
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter.Counter(targets).most_common(1)[0][0]	

print(predict(X_train, y_train, 7, L1, X_test))

# data = array-like w shape (n,m)
# targets = list containing target values (size n)
# k is the number of neighbors to consider.
# metric is a function that computes the distance between two vectors (see below).
# inputs = array-like with shape (x,m), containing x new data points each of dimension m. 
def knn(X_train, y_train, k, metric, X_test):
	predictions = []
	# # loop over all observations
	# for i in range(len(X_test)):
	# 	predictions.append(predict(X_train, y_train, k, metric, X_test.iloc[i, :]))
	# return predictions




# k_test = knn(X_train, y_train, 7, L1, X_test)

# # transform the list into an array
# k_test = np.asarray(k_test)

# # evaluating accuracy
# print("Test set score: {:.2f}".format(np.mean(k_test == y_test)))

