import pandas as pd
import numpy as np
import argparse

def get_gps(gps_file):
    X = []
    Y = []
    with open(gps_file) as f:
        gpss = f.readlines()
        for gps in gpss:
            x, y = float(gps.split()[0]), float(gps.split()[1])
            X.append(x)
            Y.append(y)
    return X, Y

def read_data_from_file(fp):
    """
    read a bunch of trajectory data from txt file
    :param fp:
    :return:
    """
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[int(t) for t in tmp]]
    return np.asarray(dat, dtype='int64')


def array_2_df(data, seq_len):
    """Converts input array to pandas dataframe"""
    lon = []
    lat = []
    tid=[]
    for traj, j in zip(data, range(1, len(data)+1)):
        for i in range(seq_len):
            lat.append(X[traj[i]])
            lon.append(Y[traj[i]])
            tid.append(j)

    converted_data = pd.DataFrame([], columns=['tid', 'lat', 'lon'])
    converted_data['tid'] = tid
    converted_data['lat'] = lat
    converted_data['lon'] = lon
    return converted_data
    
def df_2_npy(df, tid_col, seq_len):
    """Converts input panda dataframe to one-hot-encoded Numpy array (locations are still in float)."""
    
    x = [[] for i in ['lat_lon', 'day', 'hour', 'category', 'mask', '']]
    for tid in df[tid_col].unique():
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, ['lat', 'lon']].values)
        loc_list = []
        for i in range(0, len(traj)):
            lat = traj['lat'].values[i]
            lon = traj['lon'].values[i]
            loc_list.append(np.array([lat, lon], dtype=np.float64))
        x[0].append(loc_list)
        x[1].append(np.eye(seq_len-1, 7))
        x[2].append(np.eye(seq_len-1, 24))
        x[3].append(np.eye(seq_len-1, 10))
        x[4].append(np.ones(shape=(seq_len-1,1)))
    converted_data = np.array([np.array(f) for f in x])
    converted_data = converted_data[0:5]
    return converted_data
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="real.data")
    parser.add_argument("--save_path", type=str, default="train_latlon.csv")
    parser.add_argument("--tid_col", type=str, default="tid")
    parser.add_argument('-d', '--dataset', type=str, default='porto')
    args = parser.parse_args()
    
    dataset = args.dataset
    seq_len = {'geolife': 48, 'porto':40}
    X, Y = get_gps(f'data/{dataset}/gps')
    data_real = read_data_from_file(f'data/{dataset}/{args.load_path}')
    data_test = read_data_from_file(f'data/{dataset}/test.data')
    
    converted_real = array_2_df(data_real, seq_len[dataset])
    converted_test = array_2_df(data_test, seq_len[dataset])
    
    lat_centroid = (converted_real['lat'].sum() + converted_test['lat'].sum())/(len(converted_real)+len(converted_test))
    lon_centroid = (converted_real['lon'].sum() + converted_test['lon'].sum())/(len(converted_real)+len(converted_test))
    
    converted_real.to_csv(f'data/{dataset}/{args.save_path}', index=False)
    converted_test.to_csv(f'data/{dataset}/test_latlon.csv', index=False)
    
    converted_real['lat'] = (converted_real['lat'] - lat_centroid)/lat_centroid
    converted_real['lon'] = (converted_real['lon'] - lon_centroid)/lon_centroid

    converted_test['lat'] = (converted_test['lat'] - lat_centroid)/lat_centroid
    converted_test['lon'] = (converted_test['lon'] - lon_centroid)/lon_centroid
    
    converted_real.to_csv(f'data/{dataset}/dev_train_encoded_final.csv', index=False)
    converted_test.to_csv(f'data/{dataset}/dev_test_encoded_final.csv', index=False)
    
    converted_real_np = df_2_npy(converted_real, args.tid_col, seq_len[dataset])
    converted_test_np = df_2_npy(converted_test, args.tid_col, seq_len[dataset])

    np.save(f'data/{dataset}/final_train.npy', converted_real_np)
    np.save(f'data/{dataset}/final_test.npy', converted_test_np)
    np.save(f'data/{dataset}/train_encoded.npy', converted_real_np)
    np.save(f'data/{dataset}/test_encoded.npy', converted_test_np)