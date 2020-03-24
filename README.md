# Machine-Learning

# Project 1: K-Nearest Neighbor:
The goal of this project is to implement the k Nearest Neighbors classifier.

# Instructions:
Write a function in Python 3 with the following signature:

knn(data, targets, k, metric, inputs)
where
- data is an array-like with shape (n,m), containing n data points each of dimension m.
- targets is a list of size n containing the target values for the data points in data.
- k is the number of neighbors to consider.
- metric is a function that computes the distance between two vectors (see below).
- inputs is an array-like with shape (x,m), containing x new data points each of dimension m.

This function should return a list of length x containing the estimated targets for the data points in inputs, using the k nearest neighbors method.

Additionally, write functions euclidean and manhattan, which compute the Euclidean distance (also called L2 norm) and the Manhattan distance (also called L1 norm or city-block distance), respectively, such as could be passed for the formal parameter metric in your knn function.

You should implement this algorithm yourself. You should not use sklearn.neighbors.KNeighborsClassifier or any similar available implementation. You may, however, use library functions for pieces of your implementation as long as they do not violate the spirit of an assignment to implement KNN. As a rule of thumb, things from numpy or scipy are fair game, but things from sklearn are not (except that you may use the data sets from sklearn for testing, and you may use sklearn.neighbors.KNeighborsClassifier or something similar as a point of comparison for your implementation's performance). If you're not sure whether or not something is "fair game", please ask.
