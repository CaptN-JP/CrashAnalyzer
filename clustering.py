# Implement HDBSCAN to cluster the data and consider only the specified columns. The columns provided are Longitude and Latitude. 
# The output should provide the dataframe with new column called cluster.
import pandas as pd
import hdbscan
from spatial_viz import spatial_viz

# def clustering(df, columns_to_cluster):
#     # Convert the columns to a numpy array
#     X = df[columns_to_cluster].to_numpy()

#     # Cluster the data
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
#     cluster_labels = clusterer.fit_predict(X)

#     # Add the cluster labels to the dataframe
#     df['cluster'] = cluster_labels

#     # Plot the clusters using spatial_viz function
#     # spatial_viz(df, 'Longitude', 'Latitude', 'cluster', "Clusters of Crashes in Montgomery County, MD")

#     return df


# df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
# new_df = clustering(df, ['Longitude', 'Latitude'])

# # spatial_viz(
# #     df=new_df, 
# #     hover_name = 'Longitude', 
# #     hover_data = 'Latitude', 
# #     color_column = 'cluster', 
# #     title = "Clusters of Crashes in Montgomery County, MD"
# #     )

# print(new_df.head(50))
# ________________________________________________________________________________________________________________________

# from sklearn.cluster import DBSCAN

# def clustering(df, columns_to_cluster, eps=0.5, min_samples=5):
#     # Convert the columns to a numpy array
#     X = df[columns_to_cluster].to_numpy()

#     # Cluster the data
#     clusterer = DBSCAN(eps=eps, min_samples=min_samples)
#     cluster_labels = clusterer.fit_predict(X)

#     # Add the cluster labels to the dataframe
#     df['cluster'] = cluster_labels

#     return df


# df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
# new_df = clustering(df, ['Longitude', 'Latitude'])

# # ________________________________________________________________________________________________________________________

# PARTIAL DATA

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

def clustering(df, columns_to_cluster, eps=0.5, min_samples=10, sample_frac=1):
    # Scale the data
    scaler = StandardScaler()
    df[columns_to_cluster] = scaler.fit_transform(df[columns_to_cluster])

    # Sample a fraction of the data
    df_sample = df.sample(frac=sample_frac, random_state=42)

    # Convert the columns to a numpy array
    X = df_sample[columns_to_cluster].to_numpy()

    # Cluster the data
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(X)

    # Add the cluster labels to the dataframe
    df_sample['cluster'] = cluster_labels

    return df_sample

df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
df.drop(columns=
        ['Report Number', 'Local Case Number', 'Agency Name', 'Road Name', 
        'Cross-Street Type', 'Cross-Street Name', 'Off-Road Description', 
        'Person ID',
        'Circumstance', 'Drivers License State', 'Vehicle ID',
        'Speed Limit', 'Driverless Vehicle', 'Parked Vehicle', 
        'Equipment Problems'
       ], inplace= True
        )
df.dropna(inplace=True)
new_df = clustering(df, ['Longitude', 'Latitude'])

spatial_viz(
    df=new_df, 
    hover_name = 'Longitude', 
    hover_data = 'Latitude', 
    color_column = 'cluster', 
    title = "Clusters of Crashes in Montgomery County, MD"
    )