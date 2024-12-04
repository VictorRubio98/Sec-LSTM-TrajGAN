import pandas as pd
import sys

hotspot =[]

DATASET = sys.argv[1]
if DATASET == 'foursquare':
    gps_path = 'gps1.csv'
elif DATASET == 'geolife':
    gps_path = 'gps'
elif DATASET == 'porto':
    gps_path = 'gps'

target_path = sys.argv[2]

df1 = pd.read_csv(f"data/{DATASET}/{gps_path}", sep =' ') # gps file
df2 = pd.read_csv(target_path) # generated_data

df1.columns = ['lat', 'lon']
df2['lat'] = df2['lat'].round(3)
df2['lon'] = df2['lon'].round(3)


# Number of matching pairs between the two tables
matching_pairs = df2.merge(df1[['lat', 'lon']], on=['lat', 'lon'], how='inner')

# # Calculate the percentage
percentage = (len(matching_pairs) / len(df2)) * 100

print(f"Retention percentage: {percentage:.2f}%")


# import pandas as pd
# from shapely.geometry import LineString
# import numpy as np


# df3 = pd.read_csv('data/test_latlon.csv')

# line_dict1 = {}
# line_dict2 = {}

# # Calculating centroid from LineString
# def calculate_centroid(line_coords):
#     line = LineString(line_coords)
#     centroid = line.centroid
#     return (centroid.x, centroid.y)


# def process_data(df, line_dict):
#     for tid, group in df.groupby('tid'):
#         line_coords = [(row['lon'], row['lat']) for index, row in group.iterrows()]
#         line_dict[tid] = line_coords


# process_data(df2, line_dict1)
# process_data(df3, line_dict2)

# # Calculate centroids for each tid from both files
# centroids1 = np.array([calculate_centroid(line_dict1[tid]) for tid in line_dict1])
# centroids2 = np.array([calculate_centroid(line_dict2[tid]) for tid in line_dict2])

# # Calculate the difference between centroids using manhattan distance
# manhattan_distance = np.mean(np.sum(np.abs(centroids2 - centroids1), axis=1))

# print("Manhattan distance:")
# print(manhattan_distance)