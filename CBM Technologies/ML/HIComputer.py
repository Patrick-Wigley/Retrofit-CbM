# ---------------------------------------------- #
 # -  Main Asset Trend Discovery & HI Script  - #
 # -               By Patrick                 - #
# ---------------------------------------------- #
# Abstract ML component - Instantiate instance of AssetComputer
# For tracking mulitple assets

# --------------------------------------------------------------------------------------- #
#                                         TODO
#  - Refactor for Low-Coupled, with abstract classes for "DataHandler" or "ModelTrainer"
#  - Obtain Emerging Trends, such as "Degrading swiftly" or 
#      "x consecative anomalies discovered" 
#  - Update UE5 code to point to new save location (asset_status.json)
#  - Remove unnecessary constants
# --------------------------------------------------------------------------------------- #




from typing import Union

# Data Persistence tools
from pathlib import Path
import os
import json

# Data collection, processing & storing tools
import requests
from SECRETS import API_KEY, CHANNEL_ID
from sklearn.preprocessing import StandardScaler
import pandas


# Training & Prediction
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras import Sequential
import numpy as np

# Visualisation
import matplotlib.pyplot as plt



# CONSTANTS
JSON_OUT_DIR = "saves/assets_info/"
SEQUENCE_LEN = 20

API_MAX_RESULTS = 8000
OLDEST_DATE:str = "2002-02-28 00:00:00"

NOT_TRAINED = 1


class AssetComputer:
    """ 
        Components are abstracted into respective methods
        Included data-persistence, pulling new ThingSpeak API data & storing locally
    """
    def __init__(self, assetID:str):
        self.ASSET_NAME:str = assetID
        self.__CACHE_PATH:str = f"./cache/{assetID}"
        """Every asset stores cache to speed up processing in DT"""

        self.__cache_cloud_paths:dict = {"root": f"{self.__CACHE_PATH}/cloud_data", "data_points": f"{self.__CACHE_PATH}/cloud_data/data_points", "meta":f"{self.__CACHE_PATH}/cloud_data/meta"}
        self.__cache_model_path:str = f"{self.__CACHE_PATH}/trained_weights" # Only using one model currently

        self.__HI_JSON_File:str = f"./cache/{assetID}/asset_status"
        self.__training_info_JSON_File:str = f"{self.__cache_model_path}/Training_info.json"
        self.insights:dict = {}
        """ Containing metrics, analysis/insights """
        self.__training_info:dict = {}
        """ Contains point in dps where trainined elapsed,  """
        
        
        self.__raw_features_df:pandas.DataFrame = self.__data_collection()
        self.__training_data:Union[pandas.DataFrame, None]
        """ Populated during health data localisation (if enough datapoints) """
        self.processed_features_df:pandas.DataFrame = self.__preprocess()
        """ Standardised """
        self.trained_weights:Union[Sequential, None] = self.__train_model()
        """ If not enough training data then eq None """
        
    
        if self.trained_weights != None:
            self.__sequenced_dps:np.ndarray = self.__sequence_df(self.__raw_features_df)
            """ Sequence all dps """
            self.__MSE:pandas.Series = self.__anomaly_score()
            """ The reconstruction errors from autoencoders prediction against actual """

            self.insights = self.get_insights()

        else:
            self.insights.update({"training_status": NOT_TRAINED})


    def __data_collection(self) -> pandas.DataFrame:
        # 1. Check cache up-to-date
        if not os.path.isfile(self.__cache_cloud_paths["data_points"]):
            file_path = Path(self.__cache_cloud_paths["data_points"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f: f.close()

        cached_dps: pandas.DataFrame
        try:
            cached_dps = pandas.read_csv(self.__cache_cloud_paths["data_points"])
            cached_dps.index = cached_dps["created_at"]
            cached_dps.index = pandas.to_datetime(cached_dps.index, format="%Y-%m-%d %H:%M:%S")
            cached_dps.drop(columns=["created_at"], inplace=True)
        except pandas.errors.EmptyDataError:
            cached_dps = pandas.DataFrame()

        # 2. Download and cache the cloud data that's not cached already
        URL = f"https://thingspeak.mathworks.com/channels/{CHANNEL_ID}/feeds.json"
        API_payload:requests.Response
        if cached_dps.empty:
            API_payload = requests.get(URL, params={"api_key": API_KEY, "start":OLDEST_DATE})
        else:
            # Append additional dps on cloud to cache
            latest_cached_dp = str(cached_dps.tail(n=1).index[-1] + pandas.Timedelta(seconds=1)) # Add a second so it doesnt factor last row in request
            API_payload = requests.get(URL, params={"api_key": API_KEY, "start":latest_cached_dp, "results":API_MAX_RESULTS})

        data = API_payload.json()
        print(f"API Response: {API_payload.status_code}")
        
        # 3. If new data on cloud since last download, append to cache and dataframe
        features_df: pandas.DataFrame
        new_dps = pandas.DataFrame(data["feeds"])

        if not new_dps.empty:
            meta:dict = data["channel"]
            # get field names
            feature_fields = {}
            # 8 Fields
            for i in range(1, 8+1):
                if f"field{i}" in list(meta.keys()):
                    feature_fields.update({f"field{i}": meta[f"field{i}"]})

            new_dps.index = new_dps["created_at"]
            new_dps.index = pandas.to_datetime(new_dps.index, format="%Y-%m-%dT%H:%M:%SZ") # ThingSpeak Format

            # Rename ThingSpeak fields (field1, field2, etc) to feature names (RMS, Kurtosis, etc)
            new_dps.rename(columns=feature_fields, inplace=True)
            new_dps.drop(columns=["created_at", "entry_id"], inplace=True)
            
            # Concatenate cached & cloud datapoints
            features_df = pandas.concat([cached_dps, new_dps])
            # Update cache
            features_df.to_csv(self.__cache_cloud_paths["data_points"])

        else: # Else, just use cache
            features_df = cached_dps
        
        return features_df
        
    
    def __train_model(self) -> Union[Sequential, None]:
        # 1. Load trained weights
        CACHED_MODEL_PATH: str = self.__cache_model_path+"/Acceleration_8features.keras" # Just using the one model (accleration anomaly detection)
        CAHCED_MODEL_INFO_PATH: str = self.__training_info_JSON_File

        trained_weights: Union[tf.keras.Model, None] = None 
        
        if os.path.isfile(CACHED_MODEL_PATH):
            # Load cached training info (if exists)
            with open(CAHCED_MODEL_INFO_PATH) as f:
                self.__training_info = json.load(f); f.close()
            
            # Load cached Model
            return tf.keras.models.load_model(CACHED_MODEL_PATH)
        
        else:
            # Need to train model (if enough dps)
            # 1.1.1 Locate healthy data for training
            self.__training_data = self.__training_data_collection()   
        
            

            
            # If Training Complete
            if isinstance(self.__training_data, pandas.DataFrame):
                # 1.1.2 Save training info (use final index to slice data from end of training period onwards)
                Path(CAHCED_MODEL_INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
              
                with open(CAHCED_MODEL_INFO_PATH, "w") as f:
                    self.__training_info = {
                        "TrainingEndIdx": str(self.__training_data.index[SEQUENCE_LEN-1]),
                        "TrainingStartIdx": str(self.__training_data.index[0]),
                        "DPsLength":len(self.__training_data),
                        "RawDPs":[{str(row[0]): row[1].to_dict()} for row in self.__training_data.iterrows()]
                        }
                    json.dump(self.__training_info, f); f.close()

                # 1.2 Sequence training data
                __training_data_sequenced: np.ndarray = self.__sequence_df(self.__training_data)

                # 1.3 Train Model
                trained_weights = Sequential([
                    # Input Layer - Using ReLU as paper, if large reconstruction error or dead neurons detected tne use tanh
                    LSTM(64, activation="ReLU", input_shape=(SEQUENCE_LEN, len(self.__training_data.columns)), return_sequences=False),
                    RepeatVector(SEQUENCE_LEN), 
                    # Output Layer
                    LSTM(64, activation="ReLU", return_sequences=True),
                    TimeDistributed(Dense(len(self.__training_data.columns)))
                ])
                trained_weights.compile(optimizer="adam", loss="mse")
            
                info = trained_weights.fit(
                    __training_data_sequenced,
                    __training_data_sequenced,
                    epochs=50,
                    batch_size=32,
                    validation_split=.1, # Ratio 
                    shuffle=False
                )
                
                print(f"Loss: {np.mean(info.history['loss'])}")
                print(f"Val Loss: {np.mean(info.history['val_loss'])}")

                trained_weights.save(CACHED_MODEL_PATH)

            return trained_weights

    def __training_data_preprocess(self, training_df:pandas.DataFrame) -> pandas.DataFrame:
        pass # REMOVE

    def __training_data_collection(self) -> Union[pandas.DataFrame, None]:
        """ 
        Notes - Need to study machinery, some machines can operate very differently, or unusal events can occur unexpectedly during training period, ruining the trained autoencoder 
        Possible future work:
            -   LLM for data enrichment - supply meta data of environment, usage/workload during ML training to cater features, some preprocessing maybe (removing unusual datapoints) 
                for gathering more accurate & dynamic training material

        """


        def __euclidean_distance_recent_baseline(recent, all_previous):
            mu_recent = np.mean(recent, axis=0)
            mu_baseline = np.mean(all_previous, axis=0)
            return np.linalg.norm(mu_recent - mu_baseline)

        WINDOW = 10
        K_STABLE = 50    # How many consecative healthy bursts we need to declare the previous datapoints as heathy stable
        THRESHOLD = 0.7   # Distance Threshold - Needs to be tuned for datasets

        stable_counter = 0
        lock_index = None   # Identify last Index of a certain healthy range to slice dataframe - if this remains null, a suitable stable range was not yet found 

        automated_distancing = True # When this is switched off, the model will be trained the first K datapoints (could be like 1000 in practice)

        index = 0
        # Sliding Window
        for t in range(WINDOW, len(self.processed_features_df)):

            recent = self.processed_features_df[t-WINDOW:t]
            all_previous = self.processed_features_df[:t-WINDOW]

            if len(all_previous) < WINDOW:
                index+=1
                continue

            distance = __euclidean_distance_recent_baseline(recent, all_previous) if automated_distancing else 0

            if distance < THRESHOLD:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= K_STABLE:
                lock_index = t  #
                print(f"Healthy training data found up to index {t}")
                healthy_data = self.processed_features_df[:t]
                return healthy_data
            
            index+=1


        print(f"Still searching for healthy data: \n WINDOW = 10 \n K_STABLE = 25 \n THRESHOLD = 0.8 \n K_Stable_Found = {stable_counter}" )
        return None

        def __training_data_processing(self):
            """ 
                Chop sessions down and unify into n-hour/minute chunks 
                Therefore, the autoencoder learns patterns/trends within each n-hour/minute window 
                e.g. acceleration patterns over an hour (10 minute interval) takes 6 datapoints 
            """
            


    def __sequence_df(self, df:pandas.DataFrame) -> np.ndarray:
        seq_blocks = []
        print(df)

        for i in range(len(df.values) - SEQUENCE_LEN+1):
            seq_blocks.append(df.values[i:i+SEQUENCE_LEN])
        return np.array(seq_blocks)

    def __preprocess(self) -> pandas.DataFrame:
        # Beginning to consider machines uptimes - timestream is not perfectly 10min intervals 
        # as the machine isn't always running, or the connectivity is prevented.

        # One nice strategy to try is slicing the time-series data into "sessions".  

        # 1.optional Fill in missing dps (when machine is off) with a windowed average roughly around that day/time  
        FILL_MISSING = False
        if FILL_MISSING:
            self.__raw_features_df.index = pandas.to_datetime(self.__raw_features_df.index)
            self.__raw_features_df.index = self.__raw_features_df.index.floor("10min")
            self.__raw_features_df = self.__raw_features_df.groupby(self.__raw_features_df.index).mean()

            resampled_df = self.__raw_features_df.resample("10min").asfreq()
            rolling_df = resampled_df.rolling(window=6, min_periods=1).mean()
            filled_avg_df = rolling_df.ffill()

            print(f"Raw:\n{self.__raw_features_df}")
            
            print(f"Resampled:\n{filled_avg_df}")
            print(type(filled_avg_df.index))

        # 1. Group uptimes into sessions (when machine is off for extended period, session elapses and new sesh begins on bootup)
        time_diffs = self.__raw_features_df.index.to_series().diff()
        # If there is a gap larger then 30 mins, (missed 3 intervals) then this session is over
        session_mask = time_diffs > pandas.Timedelta("30min")
        self.__raw_features_df["session_id"] = session_mask.cumsum()

        session_lenths = self.__raw_features_df.groupby("session_id").size()
        print(session_lenths)


        # 2. Remove extreme outliers
        # Some datasets can have extreme outliers (I've found this in my collected dataset)
        # Therefore, this is to remove them.
        # The upper percentile values (.x% > e.g. .99% >) is removed (these are the large outliers)
        # STD could (not tested here yet) be used to determine how many outliers there are, 
        # and thus determine the upper percentile to remove (more outliers means lower percentile threshold to remove more common outliers in a dataset)
        UPPER_PERCENTILE_CLIP: float = .95
        for feature in self.__raw_features_df.columns:
            # Remove upper outliers for each feature
            clip_threshold:float = self.__raw_features_df[feature].quantile(UPPER_PERCENTILE_CLIP)
            self.__raw_features_df[feature] = self.__raw_features_df[feature].clip(upper=clip_threshold)

        self.__raw_features_df["session_id"].mul(10)
        self.__raw_features_df.plot()
        plt.legend()
        plt.tight_layout()
        plt.show()     

        # 3. Standardise features (Using blackboxed Sklearn)
        # Scalar must only take values of features (not time column) - Can check shape of self.raw_features_df.values, which should be n where n is amount of features being captured
        scaled_features = StandardScaler().fit_transform(self.__raw_features_df.values)
        scaled_features_df = pandas.DataFrame(scaled_features, 
                                 columns=self.__raw_features_df.columns.to_list(),
                                  index=self.__raw_features_df.index)
        
        return scaled_features_df
        


    def __anomaly_score(self) -> pandas.Series:
        """ Get MSE (reconstruction errors) """
        reconstructions = self.trained_weights.predict(self.__sequenced_dps)
        MSE = np.mean((self.__sequenced_dps - reconstructions)**2, axis=(1,2))

        return pandas.Series(MSE, index=self.__raw_features_df.index[SEQUENCE_LEN - 1:])


    def get_insights(self) -> dict:
        """ Get HI, degradation onset timestamp, trends """
        
        MSE_healthy_portion = self.__MSE.loc[self.__training_info["TrainingEndIdx"]:]
        MSE_smoothed = self.__MSE.rolling(window=5, min_periods=1).mean()
        
        mu = MSE_healthy_portion.mean()
        std = MSE_healthy_portion.std()

        # Using 68-95-99.7 rule
        # 1(68%), 2(95%) or 3(99.7%)
        P = 1
        # Same as probability function (Pr())
        anomalies = MSE_smoothed.loc[~((mu - (P*std) <= MSE_smoothed) & (MSE_smoothed <= mu + (P*std)))]

        # Therefore, the upper limit (positive threshold that determines anomalies) is: mu + (p*std)
        # Formula is "Upper Control Limit" - Three-Sigma Rule (https://www.geeksforgeeks.org/maths/68-95-99-rule/)
        threshold = mu + (P * std)
        anomalies = MSE_smoothed[MSE_smoothed > threshold]
        deg_onset_idx = -1 if len(anomalies) == 0 else anomalies.index[0]

        HI = self.__get_health(MSE_smoothed, threshold, deg_onset_idx)
        
        return {
            "HI": HI[-1],
            "HI_dps": HI,
            "DegradationOnset": deg_onset_idx,
            "Trend": "Current Trend",
            "anomalous_dps_count": len(anomalies)
        }


    def __get_health(self, MSE_smoothed, threshold, deg_onset_idx) -> np.ndarray:
        """ Get 100%-0% Health Score - Using smoothing and memorisation factors (Moving Averages) """

        # 1. Get damage signal & remove impact from dps below "damage threshold" (from 68-95-99.7 rule)
        damage = MSE_smoothed[deg_onset_idx:]
        damage[damage < threshold] = 0

        # 2. Moving Average function
        # 1 Eq no forgetting, 
        FORGET_FACTOR = .9
        for i in range(1, len(damage)):
            damage.iloc[i] = (FORGET_FACTOR * damage.iloc[i-1]) + (1 - FORGET_FACTOR) * damage.iloc[i]

        # 3. Fitting scores to 100-0 - Using damages as negated exponents on eulers const 
        damage = -damage
        damage_scaled = damage / damage.quantile(.95)

        return np.exp(damage_scaled)


    def get_asset_info(self) -> Union[dict, None]:        
        return {"Dummy data (get actual JSON file)"} \
            if self.__HI_JSON_File.exists() \
                else None


    def plot(self) -> None:
        SHOW_MSE: bool = True # For debugging inf issue with larger dataset

        fig, ax = plt.subplots(1,2, figsize=(15,6))




        # Training Data:
        training_data_used:pandas.DataFrame = self.processed_features_df.loc[self.__training_info["TrainingStartIdx"]:self.__training_info["TrainingEndIdx"]]
        ax[0].plot(training_data_used, label=training_data_used.columns)
        ax[0].set_title(f"{self.ASSET_NAME} - Training Data Used")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Health (%)")
        ax[0].tick_params(axis="x", rotation=45)
        

        ax[1].plot(self.processed_features_df, label=self.processed_features_df.columns)
        ax[1].set_title(f"{self.ASSET_NAME} - All Data")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Health (%)")
        ax[1].tick_params(axis="x", rotation=45)

        training_start_idx = training_data_used.index[0]
        training_end_idx = training_data_used.index[-1]
        print(training_start_idx)
        print(type(training_start_idx))
        
        ax[1].axvline(x=training_start_idx, color='green', linestyle='--', alpha=0.7)#, linewidth=4) 
        ax[1].axvline(x=training_end_idx, color='green', linestyle='--', alpha=0.7)#, linewidth=4) 
        


        test:pandas.Series = self.insights["HI_dps"][1:]
        # print(f"HI:\n{test}")
        # print(f"\nMSE:\n{self.__MSE}")
        print(f"Training Data:{self.processed_features_df}")

        # plt.plot(self.__MSE, label="MSE")


        plt.legend()
        plt.tight_layout()
        plt.show()



#- Example
if __name__ == "__main__":
    pillar_drill_1 = AssetComputer("PD1")
    print("\n\n")
    pillar_drill_1.plot()