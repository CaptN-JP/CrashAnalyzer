# Import preprocessed data and send it to the clustering model. Use the clustering model to generate clusters and visualize the clusters.
# For individual clusters, use fp_growth.py to find frequent patterns. 
# Create new attributes for the top N frequent patterns, perform one hot encoding and send the data to the classification model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmeans import k_means
from gnn import gnn
from consts import *
from spatial_viz import spatial_viz
from fp_growth import mine_frequent_patterns

df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')

# Run the clustering model and get the output dataframe
new_df = k_means(df[['Longitude', 'Latitude']], n_clusters=NUMBER_OF_CLUSTERS)
df["cluster"] = new_df["cluster"]

# Visualize the clusters
spatial_viz(
    data=df,
    hover_name="Injury Severity",
    hover_data=['Weather', 'Surface Condition', 'Light','Driver Substance Abuse','Driver At Fault'],
    color_column='cluster',
    title="Clusters of Crashes in Montgomery County, MD [K-Means]"
)

# Encoded Data Frames
encoded_dfs = []

# Run the fpgrowth model and get the output dataframe for each cluster
for i in range(NUMBER_OF_CLUSTERS):
    cluster_df = df[df['cluster'] == i]
    print(f"Cluster {i}:\n{len(cluster_df)}\n")
    freq_patterns = mine_frequent_patterns(cluster_df, columns_to_mine=columns_to_mine, **FPGROWTH_PARAMS)

    # Create new attributes for the top N frequent patterns or patterns above a certain support threshold and perform one hot encoding
    modifed_df = cluster_df.copy()
    for pattern in freq_patterns:
        modifed_df[pattern] = modifed_df.apply(lambda row: 1 if set(pattern).issubset(row[columns_to_mine]) else 0, axis=1)
    encoded_df = pd.get_dummies(modifed_df, columns=columns_to_mine) # one hot encoding    
    print(encoded_df.head(10))

    # Appending the encoded dataframe to the list of encoded dataframes
    encoded_dfs.append(encoded_df)

