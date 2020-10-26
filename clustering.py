from kneed import DataGenerator, KneeLocator
import pandas as pd
import scipy.cluster.hierarchy as shc

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def recipes_DBSCAN(nutrition_data):
    scaled_data = StandardScaler().fit_transform(nutrition_data)

    db = DBSCAN(eps=50, min_samples=40)
    db.fit(scaled_data)
    y_pred = db.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_data)

    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title("Clusters determined by DBSCAN", fontsize=20)
    ax.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_pred, cmap='Paired')
    ax.grid()
    fig.show()

def KMean_preprocess(nutrition_data):
    ks = []
    scaled_data = StandardScaler().fit_transform(nutrition_data)
    kmeans_kwargs = {
        "init": "random",
        "n_init": 15,
        "max_iter": 300,
        "random_state": None,
    }
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    kl = KneeLocator(
        range(1, 11), sse, curve = "convex", direction = "decreasing"
    )
    print("elobow result is: " + str(kl.elbow))
    ks.append(kl.elbow)

    silhouette_coefficients = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_data)
        score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_coefficients.append(score)
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    closest_1 = 2
    index = 2
    cluster_amount = 0
    for silhouette_coefficient in silhouette_coefficients:
        if abs(silhouette_coefficient - 1) < (closest_1 - 1):
            closest_1 = silhouette_coefficient
            cluster_amount = index
        index += 1
    print("silhouette result is: " + str(cluster_amount))
    ks.append(cluster_amount)

    return ks
def recipes_KMean(nutrition_data, k):
    scaled_data = StandardScaler().fit_transform(nutrition_data)
    kmeans_kwargs = {
        "init": "random",
        "n_init": 15,
        "max_iter": 300,
        "random_state": None,
    }
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_data)
    y_pred = kmeans.fit_predict(scaled_data)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_data)

    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title("Clusters determined by " + str(k) + "-means", fontsize=20)
    ax.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_pred, cmap='Paired')
    ax.grid()
    fig.show()

def recipes_hie(nutrition_data, algr):
    scaled_data = StandardScaler().fit_transform(nutrition_data)

    #use scipy.cluster library to get the dendrogram of the data, which helps me to determine the number of clusters
    plt.figure(figsize=(20, 7))
    plt.title("Clusters determined by hierarchy - " + algr)
    dend = shc.dendrogram(shc.linkage(scaled_data, method=algr))
    plt.show()

    #use sklearn.cluster library to group the data points into these number clusters
    cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage=algr)
    cluster.fit(scaled_data)
    y_pred = cluster.fit_predict(scaled_data)

    #use PCA method to reduce dimensions to two, and visualize
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_data)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    print(principalDf)
    
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title("Clusters determined by hierarchy - " + algr, fontsize=20)
    ax.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_pred, cmap='Paired')
    ax.grid()
    fig.show()


def main():
    csv_file = 'Indian/Indian_recipes_nutrition.csv'
    # csv_file = 'Mexican/Mexican_recipes_nutrition.csv'
    nutrition_data = np.genfromtxt(csv_file, delimiter=',', skip_header=True)
    nutrition_data = np.delete(nutrition_data, 0, 1)

    recipes_DBSCAN(nutrition_data)
    # recipes_hie(nutrition_data, 'single')
    # ks = KMean_preprocess(nutrition_data)
    # for k in ks:
    #     recipes_KMean(nutrition_data, k)





if __name__ == '__main__':
    main()
