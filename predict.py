import sys
import pandas as pd
import numpy as np

from model import LSTM_TrajGAN

from tensorflow.keras.preprocessing.sequence import pad_sequences


np.random.seed(2020)

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    eps = sys.argv[2]
    latent_dim = 100
    max_length = 144
    
    DATASET = 'porto'
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon":2,"day":7,"hour":24,"category":10,"mask":1}
    
    tr = pd.read_csv(f'data/{DATASET}/train_latlon.csv')
    te = pd.read_csv(f'data/{DATASET}/test_latlon.csv')
    
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    scale_factor=max(max(abs(tr['lat'].max() - lat_centroid),
                         abs(te['lat'].max() - lat_centroid),
                         abs(tr['lat'].min() - lat_centroid),
                         abs(te['lat'].min() - lat_centroid),
                        ),
                     max(abs(tr['lon'].max() - lon_centroid),
                         abs(te['lon'].max() - lon_centroid),
                         abs(tr['lon'].min() - lon_centroid),
                         abs(te['lon'].min() - lon_centroid),
                        ))
    
    gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor)
    
    # Test data
    dataset_path = 'final_test'
    x_test = np.load(f'data/{DATASET}/{dataset_path}.npy',allow_pickle=True)

    if DATASET=='foursquare':
        x_test = [x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5].reshape(-1,1),x_test[6].reshape(-1,1)]
        X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_test[:5]]
    elif DATASET=='geolife' or DATASET=='porto':
        x_test = [x_test[0],x_test[1],x_test[2],x_test[3],x_test[4]]
        X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_test[:5]]
    
    # Add random noise to the data
    noise_length = 1027 if DATASET == 'foursquare' else 2500 
    noise = np.random.normal(0, 1, (noise_length, 100))
    X_test.append(noise)
    
    # Load params for the generator
    gan.generator.load_weights('training_params/G_model_' + str(n_epochs) + '.h5') # params/G_model_2000.h5
    
    # Make predictions
    prediction = gan.generator.predict(X_test)
    
    traj_attr_concat_list = []
    for attributes in prediction:
        traj_attr_list = []
        idx = 0
        for row in attributes:
            if DATASET == 'geolife':
                cut = 48
            elif DATASET == 'porto':
                cut = 40
            else:
                cut = x_test[6][idx][0]
            if row.shape == (max_length, 2):
                traj_attr_list.append(row[max_length-cut:])
            else:
                traj_attr_list.append(np.argmax(row[max_length-cut:],axis=1).reshape(cut,1))
            idx += 1
        traj_attr_concat = np.concatenate(traj_attr_list)
        traj_attr_concat_list.append(traj_attr_concat)
    traj_data = np.concatenate(traj_attr_concat_list,axis=1)
    
    df_test = pd.read_csv(f'data/{DATASET}/dev_test_encoded_final.csv')
    tid = np.array(df_test['tid']).reshape(-1,1)
    if DATASET == 'foursquare':
        label = np.array(df_test['label']).reshape(-1,1)
        traj_data = np.concatenate([label,tid,traj_data],axis=1)
        df_traj_fin = pd.DataFrame(traj_data)
        df_traj_fin.columns = ['label','tid','lat','lon','day', 'hour', 'category','mask']
    elif DATASET == 'geolife' or DATASET=='porto':
        traj_data = np.concatenate([tid, traj_data],axis=1)
        df_traj_fin = pd.DataFrame(traj_data)
        df_traj_fin.columns = ['tid','lat','lon','day', 'hour', 'category','mask']
    
    
    
    
    # Convert location deviation to longtitude and latitude
    df_traj_fin['lat'] = df_traj_fin['lat'] + gan.lat_centroid
    df_traj_fin['lon'] = df_traj_fin['lon'] + gan.lon_centroid
    
    del df_traj_fin['mask']
    
    df_traj_fin['tid'] = df_traj_fin['tid'].astype(np.int32)
    df_traj_fin['day'] = df_traj_fin['day'].astype(np.int32)
    df_traj_fin['hour'] = df_traj_fin['hour'].astype(np.int32)
    df_traj_fin['category'] = df_traj_fin['category'].astype(np.int32)
    if DATASET == 'foursquare':
        df_traj_fin['label'] = df_traj_fin['label'].astype(np.int32)
    
    # Save synthetic trajectory data
    df_traj_fin.to_csv(f'results/{DATASET}/{eps}/syn_traj_test.csv',index=False)
    
    
    
    
    
    
    
    
    
    