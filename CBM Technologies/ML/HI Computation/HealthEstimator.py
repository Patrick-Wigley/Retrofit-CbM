import os
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# From external dataset (DEBUGGING)
from DatasetCollection import feature_standardise
# From API 
from APIHandle import pull_features


import json
import datetime as dt

from typing import Union, Any

DEBUG = True

DATAPOINTS_PER_FILE = 20480
SAMPLE_RATE = 20000 # 20 kHz


# Sequence length is the amount of consecutive bursts to join together for assessing (rather than just 1 individual burst)
SEQUENCE_LENGTH = 20

FEATURES = [
    # Time Domain Features_df
    "rms",
    "std",
    "ptp",
    "kurtosis",
    "skew",
    "crest",

    # Frequency Domain Features_df
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_total",
    "spectral_entropy",
    "frequency_peak"
]

# Get features_df from ThingSpeak

# Features_df
# features_df_df = feature_extraction() DONT NEED THIS NOW
# features_scaled_df = feature_standardise(features_df_df)


def identify_healthy_data(df:pd.DataFrame):
    def euclidean_distance_recent_baseline(recent, all_previous):
        mu_recent = np.mean(recent, axis=0)
        mu_baseline = np.mean(all_previous, axis=0)
        return np.linalg.norm(mu_recent - mu_baseline)


    WINDOW = 10
    K_STABLE = 25    # How many consecative healthy bursts we need to declare the previous datapoints as heathy stable
    THRESHOLD = 0.8   # Distance Threshold - Needs to be tuned for datasets

    stable_counter = 0
    lock_index = None   # Identify last Index of a certain healthy range to slice dataframe - if this remains null, a suitable stable range was not yet found 

    automated_distancing = False # When this is switched off, the model will be trained the first K datapoints (could be like 1000 in practice)

    index = 0
    # Sliding Window
    for t in range(WINDOW, len(df)):

        recent = df[t-WINDOW:t]
        all_previous = df[:t-WINDOW]

        if len(all_previous) < WINDOW:
            index+=1
            continue

        distance = euclidean_distance_recent_baseline(recent, all_previous) if automated_distancing else 0

        if distance < THRESHOLD:
            stable_counter += 1
            if DEBUG:
                print(f"[{stable_counter}]: {distance}")
        else:
            stable_counter = 0

        if stable_counter >= K_STABLE:
            lock_index = t
        
        index+=1

    if lock_index != None:
        print(f"Healthy training data found up to index {lock_index}")

        healthy_data = df[:lock_index]
        return healthy_data
    else:
        print(f"Still searching for healthy data: \n WINDOW = 10 \n K_STABLE = 25 \n THRESHOLD = 0.8 \n K_Stable_Found = {stable_counter}" )
        return None
    

def get_sequences(df):
    # Sequencing (using sliding window)
    data = df[list(FEATURES)].values
    sequences = []
    for i in range(len(data) - SEQUENCE_LENGTH + 1):
        sequences.append(data[i:i+SEQUENCE_LENGTH])
    
    return np.array(sequences)



def train_model(healthy_sequences, save=True):
    features_count_df = len(FEATURES)

    autoencoder = Sequential([
        # Input Layer - Using ReLU as paper, if large reconstruction error or dead neurons detected tne use tanh
        LSTM(64, activation="ReLU", input_shape=(SEQUENCE_LENGTH, features_count_df), return_sequences=False),
        RepeatVector(SEQUENCE_LENGTH), 
        # Output Layer
        LSTM(64, activation="ReLU", return_sequences=True),
        TimeDistributed(Dense(features_count_df))
    ])

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.summary() 
    
    history = autoencoder.fit(
        healthy_sequences,
        healthy_sequences,
        epochs=50,
        batch_size=32,
        validation_split=.1, # Ratio 
        shuffle=False
        )

    print(f"Loss: {np.mean(history.history['loss'])}")
    print(f"Val Loss: {np.mean(history.history['val_loss'])}")


    # Save Model
    if save:
        autoencoder.save("./saves/models/live_autoencoder.keras")

    return autoencoder



def get_reconstruction_errs(autoencoder, all_sequences, df, healthy_df):
    reconstruction = autoencoder.predict(all_sequences)
    reconstruction_error = np.mean((all_sequences - reconstruction)**2, axis=(1,2)) 
    
    print(len(reconstruction), len(all_sequences), len(reconstruction_error))


    # Errors aligned with original timeline (bursts times) NOTE - First burst doesnt have an error (error is corresponds to last burst of each sequence)
    error_series = pd.Series(
        reconstruction_error,
        index=df.index[SEQUENCE_LENGTH - 1:]
    )

    error_series.plot()

    healthy_error = error_series.loc[healthy_df.index[SEQUENCE_LENGTH-1:]]
    # Smoothing ensures that anomaly occurances persist over more than one burst (window=5)
    # Thus, good for reducing false positives
    smoothed_error = error_series.rolling(window=5, min_periods=1).mean()

    # plt.legend()
    # plt.show()

    mu = healthy_error.mean() # mu μ is terminology in math for mean
    sigma = healthy_error.std() # Sigma σ is terminology in math for std 


    # Anomalies are those with errors above threshold
    # Using 68-95-99.7 rule
    # p=  1(68%), 2(95%) or 3(99.7%)
    P = 1
    # Same as probability function (Pr())
    anomalies = error_series.loc[~((mu - (P*sigma) <= error_series) & (error_series <= mu + (P*sigma)))]
    print(f"Abnormal Values found: {len(anomalies)} ({len(error_series)} values in total)")

    # Therefore, the upper limit (positive threshold that determines anomalies) is: mu + (p*sigma)
    # Formula is "Upper Control Limit" - Three-Sigma Rule (https://www.geeksforgeeks.org/maths/68-95-99-rule/)
    threshold = mu + (P * sigma)
    anomalies = error_series[error_series > threshold]
    first_abnormal_idx = -1 if len(anomalies) == 0 else anomalies.index[0]

    #smoothed_error.plot()

    return smoothed_error, threshold, first_abnormal_idx


def get_health_index(smoothed_error, threshold, first_abnormal_idx):
    d_signal = smoothed_error[first_abnormal_idx:] # np.maximum(smoothed_error - threshold, 0)
    d_signal[d_signal < threshold] = 0


    d_signal.plot()
    plt.show()

    # This allows the damage score to come back down when error shrink
    # If this is 1 then there is no forgetting factor (1) acts as a cumsum
    _lambda = .9

    # Note, remember when you do REVERSE_TIME = anything but 0, it is leaving the rest to be set to
    # 0.0 as above defines a series of 0.0s initially.
    REVERSE_TIME = 0 # n datapoints to chop off end
    damage = pd.Series(0.0, index=d_signal.index[:-1-(REVERSE_TIME-1 if REVERSE_TIME != 0 else 0)]) # Chopping n datapoints off the tail (end)

    print(len(damage))

    # (1) Damage is accumalative 
    for n in range(1, len(damage)):

        # Unscaled exponentially weighted sum
        # Same result as _lambda * damage! - Exponentially Weighted Moving Average?
        # Forgetting factor brings value back down when errors shrink
        damage.iloc[n] = (_lambda * damage.iloc[n-1]) + (1 - _lambda) * d_signal.iloc[n]


    # Damage should come down as errors accumalate (invert inreasing damage to decreasing)
    damage = -damage
    print(damage)


    # Scaling by the upper/larger values around the .95% mark
    damage_scaled = damage / d_signal.quantile(.95)

    # Exponential normalisation i think (ensures values stay within 1-0) 
    HI = np.exp(damage_scaled)

    plt.plot(HI)

    return HI[-1], HI


def write_status(status:dict = None) -> None:
    if status:
        # Add current time
        status.update({"time": dt.datetime.now().strftime("%d/%m/%y - %H:%M")})

        with open("asset_status.json", "w") as f:
            # Dump dict into JSON - Serialised
            json.dump(status, f)
    else:
        return


VERBOSE = False

if __name__ == "__main__":
    # DEBUGGING
    with open("ScriptExecuted.txt", "w") as f:
        f.write(str(dt.datetime.now()))
        f.close()

    # Require states
    # Learning - Finding healthy stable data to train model 
    # Healthy - Model has been trained and data is not anomalous
    # Degrading - degradation period has began - begin EWMA HI 


    #- Data Preparation & Collection -# - Collected from Thingspeak channel(s) (within APIHandle.py)
    features_df:pd.DataFrame = pull_features()
    # Get features obtained from pull_features(...) - These will be the fields stored on the ThingSpeak channel(s)
    FEATURE_NAMES:list = features_df.columns.to_list() 
    # Standardisation ensures features are brought into a simililar & fair range - ensures STD is 1 as discussed in methodology
    stand_df:pd.DataFrame = feature_standardise(features_df, FEATURE_NAMES)
    
    autoencoder:tf.keras.Model = None

    if "live_autoencoder.keras" in os.listdir("./saves/models/"): # and "status_variables.json" in os.listdir("./saves/"):
        # Machines trained model exists on disk, can use this to continue assessing streaming data from IIoT component
        autoencoder = keras.models.load_model("./saves/models/live_autoencoder.keras")
        
    else:
        # Capture machines patterns over x amount of time to use for training
        healthy_df:Union[pd.Series, None] = identify_healthy_data(stand_df)         
        if isinstance(healthy_df, (pd.DataFrame, pd.Series)):
            # Ready to train
            # Healthy sequences - populated with the machines observed patterns since first applying the sensor (defined as |HS| in methodology)
            healthy_sequences = get_sequences(healthy_df)
            autoencoder = train_model(healthy_sequences)
        else:
            print("Still Capturing Machines Patterns")
            # Else, still require more time for the model to learn the machines patterns 
            write_status({"HI":"-", "ModelStatus":"Still Learning Machines Stable Healthy Pattterns", "DatapointsCollected": len(features_df)})

    if autoencoder:
        #- Anomaly Scroring and Health Estimation -#
        all_sequences = get_sequences(features_df)
        # Compute MSE_i, threshold as discussed in methodology/Phase 3/Machine Learning & HI Computation/HI Estimation
        smoothed_error, threshold, first_abnormal_idx = get_reconstruction_errs(autoencoder, (all_sequences).astype("float32"), features_df, healthy_df)
        
        if VERBOSE: smoothed_error.plot()
        
        #- Health Index Estimation -# - Compute HI from damage scores (MSE)
        current_HI, _ = 1 if first_abnormal_idx == -1 else get_health_index(smoothed_error, threshold, first_abnormal_idx)
        print("Current HI: ", current_HI)
        write_status({"HI":current_HI, "ModelStatus":"Scoring Anomalies"})

     




if False:
    class HealthIndexEstimator:
        def __init__(self, layers:list, data):
            #super.__init__(layers)
            self.sequential_layers:list = layers


            self.MODELS_LOCATION = "./models/"
            self.dataset:pd.DataFrame = pd.DataFrame(data)



        def construct(self):
            pass

        def feature_extraction(self, csv_delemiter="\t"):
            pass





    SEQUENCE_LENGTH = 20
    FEATURES_COUNT_df = 11

    autoencoder = Autoencoder([
        # Input Layer - Using ReLU as paper, if large reconstruction error or dead neurons detected tne use tanh
        LSTM(64, activation="ReLU", input_shape=(SEQUENCE_LENGTH, FEATURES_COUNT_df), return_sequences=False),
        RepeatVector(SEQUENCE_LENGTH), 
        # Output Layer
        LSTM(64, activation="ReLU", return_sequences=True),
        TimeDistributed(Dense(FEATURES_COUNT_df))
    ]
    , data=[]
    )

    # Import Dataset
    DATASET_LOC = "./Datasets/"
    NASA_BEARING_DATASET_LOC = DATASET_LOC + "NASABearing/"
    DEBUG = True

    # For all sets_directories (not files) in dataset dir - Path.join is concatenates full path str lit to directory
    sets_directories = [os.path.join(NASA_BEARING_DATASET_LOC, d)+f"/{d}/" for d in  os.listdir(NASA_BEARING_DATASET_LOC) if os.path.isdir(os.path.join(NASA_BEARING_DATASET_LOC, d))]

    ## Select Dataset ##
    DATASET_DIR_INUSE = sets_directories[0]

    if DEBUG:
        print("Available sets in Dataset are:")
        for i, dir in enumerate(sets_directories):
            print(f"[{i}] {dir}")
        # Using set/run 2
        print(f"Using set: {DATASET_DIR_INUSE}")


    print(f"Count of bursts/files: {len(os.listdir(DATASET_DIR_INUSE))}")