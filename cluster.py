import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import wrangle, scale
from sklearn.cluster import KMeans


def show_cluster(X, clusters, cluster_name, size=None, hide=False):
    '''
    use scaled data when using show cluster, as it assumes scaleing. 
    return a scatter plot of the clusters and the updated dataframe passed in to include the clusters
    '''
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X)
    X[cluster_name] = kmeans.predict(X)
    if hide == False:
        plt.figure(figsize=(16,9))
        plt.title('{} VS {}'.format(X.columns[0], X.columns[1]))
        sns.scatterplot(x= X.columns[0], y= X.columns[1], data = X, hue = cluster_name, size=size, sizes = (5,50))
        plt.show()
    return X
   
    
def view_intertia(X):
    plt.figure(figsize=(10,4))
    pd.Series({k:KMeans(k).fit(X).inertia_ for k in range(2,12)}).plot(marker='P')
    plt.xticks(range(2,12))
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title("Intertia as k increases")
    plt.show()
    
def map_clusters(df, cluster_name, size=None):
    '''
    maps longitude and latitude with the hue of whatever cluster you pass in as argument
    '''
    plt.figure(figsize=(16,9))
    plt.title('{} Map'.format(cluster_name))
    sns.scatterplot(x='latitude', y= 'longitude', data = df, hue = cluster_name, size=size, sizes = (10,50))
    plt.show()

    

