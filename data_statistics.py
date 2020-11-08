import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


def show_header(csv_file):
    df = pd.read_csv(csv_file)
    chosen_nutritions_with_cal = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    chosen_nutritions_without_cal = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    nutrients = []
    for nutrient in chosen_nutritions_without_cal:
        nutrients.append(df.columns[nutrient+1])
    print("nutrients list: " + str(nutrients))


def recipes_heatmap(nutrition_data):
    g = sns.clustermap(nutrition_data, standard_scale=1, method="average")
    plt.title('heatmap_average_method')
    plt.show()


def mean_std(clusters, chosen_data_with_calor, k):
    clusters = np.array(clusters)
    clusters = clusters.reshape((len(clusters), 1))
    data_clustered = np.append(chosen_data_with_calor, clusters, 1)
    print(data_clustered)
    for type in range(0, k):
        ind = np.squeeze(np.asarray(clusters[:, -1])) == type
        print("cluster_" + str(type))
        subcluster = data_clustered[ind, :]
        print("mean: " + str(np.mean(subcluster[:, 0])))
        print("standard variance: " + str(np.std(subcluster[:, 0])))


def frequency_distribution(clusters, chosen_data_with_calor, k):
    clusters = np.array(clusters)
    clusters = clusters.reshape((len(clusters), 1))
    data_clustered = np.append(chosen_data_with_calor, clusters, 1)
    print(data_clustered)
    frequency_data = []
    for type in range(0, k):
        ind = np.squeeze(np.asarray(clusters[:, -1])) == type
        subcluster = data_clustered[ind, :]
        frequency_data.append(subcluster)

    plt.figure(figsize=(10, 7), dpi=80)
    for cluster in frequency_data:
        label = cluster[0][-1]
        sns.distplot(cluster[:, 0], hist=False, kde=True,
                     kde_kws={'linewidth': 4},
                     label=label)
    plt.xlim(0, 1000)
    plt.legend()
    plt.show()


def linear_regression(csv_file, chosen_data_with_calor):
    show_header(csv_file)
    y = chosen_data_with_calor[:, 0]
    x = np.delete(chosen_data_with_calor, 0, 1)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    for item in abs(model.coef_):
        print(item)
