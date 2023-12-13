# Use K-Means Cluster to cluster the data based on Longitude and Latitude.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from spatial_viz import spatial_viz

def k_means(data, n_clusters):
    '''
    This function is used to cluster the data based on longitude and latitude.
    Input: data: the data to be clustered
        n_clusters: the number of clusters
    Output: the dataframe with a new column 'cluster' which contains the cluster labels of each data point
    '''
    # Standardize the data
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    # Use K-Means to cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_std)
    # Get the cluster labels
    labels = kmeans.labels_
    # Add cluster labels as a new column to the dataframe
    data['cluster'] = labels
    return data  # return the entire dataframe instead of just labels

df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
new_df = k_means(df[['Longitude', 'Latitude']], 10)
df["cluster"] = new_df["cluster"]
print(new_df.head(50))

spatial_viz(
    df=df, 
    # hover_name = 'Longitude', 
    hover_name = "Municipality",
    hover_data = 'Latitude', 
    color_column = 'cluster',
    title = "Clusters of Crashes in Montgomery County, MD"
    )

