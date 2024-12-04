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