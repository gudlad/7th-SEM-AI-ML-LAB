# A GMM is an unsupervised clustering technique based on
#  probability density estimations using the
#  Expectation-Maximization algorithm.

# Gaussian Mixture Models (GMMs)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans  # UnSupervised Learning ALgorithm
from sklearn import preprocessing  # converts raw data to preprocessed data
from sklearn import datasets

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import some data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
print('raw X', X)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length',
             'Petal_Width']  # * give labels for the columns

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

# *******************************************************
# * Visualize the clustering results
plt.figure(figsize=(14, 14))
colormap = np.array(['red', 'lime', 'black'])

# * Plot the original classifications using Petal features
# The subplot() function takes three arguments that describes the layout of the figure.
# First argument specifies number of rows.
# Second argument specifies number of columns.
# Third argument represents the index of the current plot.

plt.subplot(2, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
# target values : 0,1,2
plt.title('Real Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Build the K Means model
model = KMeans(n_clusters=3)
model.fit(X)    # model.labels_ : Gives cluster no for which samples belongs to


# * Plot the KNN model classification
plt.subplot(2, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
# model.labels_ : Gives cluster no for which samples belongs to
plt.title('K-means Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# General EM for GMM
# transform your data such that its distribution will have a
# mean value 0 and standard deviation of 1.

# * data preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)
# # StandardScaler standardize features by removing the mean and scaling to unit variance.
# # transform data such that its distribution will have a mean value 0 and standard deviation of 1.

gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)

# * plot the gmm model classification
plt.subplot(2, 2, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_y], s=40)
plt.title('GMM Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

print("Observation:The GMM using EM algorithm based clustering matched the true labels more closely than Kmeans")
plt.show()
