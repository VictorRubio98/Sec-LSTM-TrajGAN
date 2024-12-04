# Sec-LSTM-TrajGAN

## Datasets

* **GeoLife**: This GPS trajectory dataset was collected by the MSRA GeoLife project with 182 users in a period of over five years.  
  * Link: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F
* **Porto Taxi**: This GPS tarjectory dataset was collected during one year in the city of Porto, in Portugal. The dataset contains taxi trips that are sampled every 15 seconds to form a sequence of points.
    * Link: https://figshare.com/articles/dataset/Porto_taxi_trajectories/12302165?file=22677902

## Requirements

* **Python = 3.7.12** 
* **numpy = 1.19.5** 
* **scipy = 1.7.3** 
* **pandas = 1.3.5**
* **tensorflow = 2.4.0** 
* **tensorflow-privacy = 0.6.1** 
* **scikit-learn=0.23.2**
* **Keras=2.2.4**
* **geohash2=1.1**

## Usage

# Process data:

`python data/{dataset_name}/data2csv.py --load_path {file_name} --save_path train_latlon.csv`

Where `dataset_name` is the name of the dataset to use: `foursquare`, `geolife` or `porto`, `file_name` is the current name of the file to load the data from, and `save_path` is the name of the csv file to save on.

# Training:
Open `train.py`and change the parameters needed to train with the desired privacy budget, then run:

`python train.py 2000 256 100`

Where `2000` is the total training epochs, `256` is the batch size, `100` is the parameter saving interval (i.e., save params every 100 epochs).

# Prediction:
Generate synthetic trajectory data based on the real test trajectory data and save them to `results/{dataset}/syn_traj_test.csv`.

`python predict.py 2000`

Where `1900` means we load the params file saved at the 1900th epoch to generate synthetic trajectory data. Beware that if the model is trained for 2000 epochs with a desired epsilon, and the prediction is made with a save of the model from a previous epoch, the resulting data will not have the desired epsilon.

# Evaluation:

**Utility:**

`eval1.py` handles the utility metrics for statistical similarity:

`python eval1.py {test_dataset_path} {synthetic_dataset_path}`

`eval2.py` handles the utility metrics for point retention:

`python eval2.py {synthetic_dataset_path}`


**Privacy**

The privacy evaluation stated bellow comes from the oriignal article of LSTM-TrajGAN. 
Evaluate the synthetic trajectory data on the Trajectory-User Linking task using MARC.

`python TUL_test.py data/{dataset}/train_latlon.csv results/{dataset}/syn_traj_test.csv 100`

Where `data/{dataset}/train_latlon.csv` is the training data, `results/{dataset}/syn_traj_test.csv` is the synthetic test data, `100` is the embedder size.