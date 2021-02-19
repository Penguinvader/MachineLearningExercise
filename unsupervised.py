import numpy as np  # Numerical Python library
from matplotlib import pyplot as plt  # Matlab-like Python module
from urllib.request import urlopen  # importing url handling
from sklearn.cluster import KMeans  # importing clustering algorithms
# function for Davies-Bouldin goodness-of-fit
from sklearn.metrics import davies_bouldin_score
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
import urllib.request

conversions = {
    b'Female': 0,
    b'Male': 1,
    b'no': 0,
    b'yes': 1,
    b'Sometimes': 1,
    b'Frequently': 2,
    b'Always': 3,
    b'Automobile': 0,
    b'Motorbike': 1,
    b'Bike': 2,
    b'Public_Transportation': 3,
    b'Walking': 4,
    b'Insufficient_Weight': 0,
    b'Normal_Weight': 1,
    b'Overweight_Level_I': 2,
    b'Overweight_Level_II': 3,
    b'Obesity_Type_I': 4,
    b'Obesity_Type_II': 5,
    b'Obesity_Type_III': 6
}


def cvt(x): return conversions[x]


with urllib.request.urlopen("https://raw.githubusercontent.com/Penguinvader/MLFiles/main/ObesityDataSet_raw_and_data_sinthetic.csv") as file:
    attribute_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC',
                       'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']
    data = np.loadtxt(file, delimiter=",", skiprows=1, converters={
                      0: cvt, 4: cvt, 5: cvt, 8: cvt, 9: cvt, 11: cvt, 14: cvt, 15: cvt, 16: cvt})


X = data[:, :-1]  # input attributes
y = data[:, -1]  # label attribute
n = X.shape[0]  # number of records
p = X.shape[1]  # number of attributes
k = y.shape[0]  # number of target classes

feature_selection = SelectKBest(k=2)
feature_selection.fit(X, y)
scores = feature_selection.scores_
features = feature_selection.transform(X)
mask = feature_selection.get_support()
feature_indices = []
for i in range(p):
    if mask[i] == True:
        feature_indices.append(i)
x_axis, y_axis = feature_indices

print('Importance weight of input attributes')
for i in range(p):
    print(attribute_names[i], ': ', scores[i])
fig = plt.figure(1)
plt.title('Scatterplot for obesity dataset, dimension reduction with SelectKBest')
plt.xlabel(attribute_names[x_axis])
plt.ylabel(attribute_names[y_axis])
plt.scatter(X[:, x_axis], X[:, y_axis], s=50, c=y)
plt.show()

pca = PCA(n_components=2)
pca.fit(X)
obesity_pc = pca.transform(X)

fig = plt.figure(2)
plt.title('Dimension reduction of the Iris data by PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(obesity_pc[:, 0], obesity_pc[:, 1], s=50, c=y)
plt.show()

K = 5

# K-means clustering with fix K
kmeans_cluster = KMeans(n_clusters=K, random_state=2020)
kmeans_cluster.fit(obesity_pc)  # fiting cluster model for X
y_pred = kmeans_cluster.predict(obesity_pc)  # predicting cluster label
# sum of squares of error (within sum of squares)
sse = kmeans_cluster.inertia_
centers = kmeans_cluster.cluster_centers_  # centroid of clusters

# Davies-Bouldin goodness-of-fit
DB = davies_bouldin_score(obesity_pc, y_pred)

# Printing the results
print(f'Number of cluster: {K}')
print(f'Within SSE: {sse}')
print(f'Davies-Bouldin index: {DB}')

# Visualizing of datapoints with cluster labels and centroids
fig = plt.figure(3)
plt.title('Scatterplot of datapoints with clusters')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(obesity_pc[:, 0], obesity_pc[:, 1], s=50,
            c=y_pred)  # dataponts with cluster label
plt.scatter(centers[:, 0], centers[:, 1], s=50, c='red')  # centroids
plt.show()

# Finding optimal cluster number
Max_K = 31  # maximum cluster number
SSE = np.zeros((Max_K-2))  # array for sum of squares errors
DB = np.zeros((Max_K-2))  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2
    kmeans = KMeans(n_clusters=n_c, random_state=2020)
    kmeans.fit(X)
    y_pred = kmeans.labels_
    SSE[i] = kmeans.inertia_
    DB[i] = davies_bouldin_score(X, y_pred)

    # Visualization of SSE values
fig = plt.figure(4)
plt.title('Sum of squares of error curve')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.plot(np.arange(2, Max_K), SSE, color='red')
plt.show()

# Visualization of DB scores
fig = plt.figure(5)
plt.title('Davies-Bouldin score curve')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(np.arange(2, Max_K), DB, color='blue')
plt.show()
