import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def show_header(csv_file):
    df = pd.read_csv(csv_file)
    chosen_nutritions_with_cal = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    chosen_nutritions_without_cal = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    nutrients = []
    for nutrient in chosen_nutritions_without_cal:
        nutrients.append(df.columns[nutrient+1])
    #print("nutrients list: " + str(nutrients))

    return nutrients




def mean_std(clusters, chosen_data_with_calor, k):
    mean = []
    clusters = np.array(clusters)
    clusters = clusters.reshape((len(clusters), 1))
    data_clustered = np.append(chosen_data_with_calor, clusters, 1)
    print(data_clustered)
    for type in range(0, k):
        ind = np.squeeze(np.asarray(clusters[:, -1])) == type
        print("cluster_" + str(type))
        subcluster = data_clustered[ind, :]
        mean.append(np.mean(subcluster[:, 0]))
        print("mean: " + str(np.mean(subcluster[:, 0])))
        print("standard variance: " + str(np.std(subcluster[:, 0])))
    return mean
def cluster_index_with_nutr_data(clusters, chosen_data_with_calor):
    clusters = np.array(clusters)
    clusters = clusters.reshape((len(clusters), 1))
    data_clustered = np.append(chosen_data_with_calor, clusters, 1)
    return data_clustered


def linear_regression(csv_file, chosen_data_with_calor):
    print("nutrients list: " + str(show_header(csv_file)))

    y = chosen_data_with_calor[:, 0]
    x = np.delete(chosen_data_with_calor, 0, 1)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    for item in model.coef_:
        print(item)

def correlation_diagram_2(csv_file, chosen_data_with_calor, index):
    header = show_header(csv_file)
    y = chosen_data_with_calor[:, 0]
    x = chosen_data_with_calor[:, index]
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(header[index-1], fontsize=15)
    ax.set_ylabel('calories', fontsize=15)
    ax.scatter(x,y, cmap='Paired')
    ax.grid()
    fig.show()

def spearman_correlation(chosen_data_with_calor, x_index, y_index):
    header = ['calories', 'TotalFat', 'SaturatedFat', 'Cholesterol', 'Sodium', 'Potassium', 'TotalCarbohydrates',
              'Protein', 'Sugars', 'VitaminA']
    y = chosen_data_with_calor[:, y_index]
    x = chosen_data_with_calor[:, x_index]
    corr, _= spearmanr(x, y)
    print(header[y_index] + '/' +  header[x_index] + ' Spearmans correlation: %.3f' % corr)

def frequency_distribution(clusters, chosen_data_with_calor, k):
    clusters = np.array(clusters)
    clusters = clusters.reshape((len(clusters), 1))
    data_clustered = np.append(chosen_data_with_calor, clusters, 1)
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

def visualization_spearman_3D(chosen_data_with_calor, x_index, y_index):

    header = ['calories','TotalFat','SaturatedFat','Cholesterol','Sodium','Potassium','TotalCarbohydrates','Protein','Sugars', 'VitaminA']
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")
    # ax = fig.add_subplot()
    x = chosen_data_with_calor[:, x_index]
    y = chosen_data_with_calor[:, y_index]
    z = chosen_data_with_calor[:, 0]

    X, Y = np.meshgrid(x, y)
    mean_z = np.mean(z)
    Z = np.zeros(X.shape)
    for i in range(len(Z)):
        Z[i] = np.array([mean_z] * len(Z))
    for i in range(len(Z)):
        Z[i][i] = z[i]



    # ax.set_xlabel(header[x_index])
    # ax.set_ylabel(header[y_index])
    # levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
    # cmap = plt.get_cmap('PiYG')
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # cf = ax.contourf(X, Y, Z, levels=levels,
    #                   cmap=cmap)
    # fig.colorbar(cf, ax=ax)
    # plt.show()

    # Creating color map
    my_cmap = plt.get_cmap('Set2')

    #surf = ax.plot_trisurf(x, y, z, cmap="rainbow")
    surf = ax.scatter3D(x, y, z,  alpha = 0.8,
                    c = (x + y + z),
                    cmap = my_cmap,
                    marker ='o')

    fig.colorbar(surf,ax = ax, shrink=0.5, aspect=10)
    ax.set_xlim(0, 150)
    ax.set_xlabel(header[x_index])
    ax.set_ylim(0, 150)
    ax.set_ylabel(header[y_index])
    plt.show()








