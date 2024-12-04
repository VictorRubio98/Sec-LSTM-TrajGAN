import sys
import pandas as pd
import numpy as np

from model import LSTM_TrajGAN
import tensorflow as tf
import tensorflow_privacy
from argparse import ArgumentParser


if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    n_batch_size = int(sys.argv[2])
    n_sample_interval = int(sys.argv[3])
    
    ## Parameters to modify
    DATASET = 'porto' #Can be porto, geolife or foursquare
    noise_multiplier = 0 # Amount of noise to add in DP
    max_grad = 30 # Clipping value of the gradients
    lr = 0.001
    private = True # Train with DP-SGD or not
    
    latent_dim = 100
    max_length = 144


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

    gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor, private=private, noise_multiplier=noise_multiplier, max_grad=max_grad, lr = lr)
    
    
    gan.train(epochs=n_epochs, batch_size=n_batch_size, sample_interval=n_sample_interval, dataset=DATASET)

    if private and noise_multiplier != 0:
        tensorflow_privacy.compute_dp_sgd_privacy(
            n = len(tr), 
            batch_size = n_batch_size, 
            epochs = n_epochs, 
            noise_multiplier = noise_multiplier,
            delta = 1e-8)