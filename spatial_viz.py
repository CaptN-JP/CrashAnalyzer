import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
SHAPES_FILE = 'data/street_centerline_montgomery/street_centerline.shp'

def spatial_viz(df, shp_file):
    # Create a geopandas dataframe from the .shp file
    gdf = gpd.read_file(shp_file)

    # Create a geopandas dataframe from the df
    gdf2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))

    # Create a scatter plot using plotly
    # fig = px.scatter_mapbox(gdf2, lat='Latitude', lon='Longitude', hover_name='Injury Severity', color_discrete_sequence=['red'], zoom=10, hover_data=['Weather', 'Surface Condition', 'Light','Driver Substance Abuse','Driver At Fault'],)
    fig = px.scatter_mapbox(gdf2, lat='Latitude', lon='Longitude', 
                            hover_name='Cluster', 
                            color="Cluster", 
                            color_discrete_sequence=px.colors.qualitative.Plotly, 
                            zoom=10, 
                            hover_data=['Weather', 'Surface Condition', 'Light','Driver Substance Abuse','Driver At Fault'])
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(title="Crashes in the County of Montgomery, State of Maryland, United States")
    fig.show()

spatial_viz(df, SHAPES_FILE)
