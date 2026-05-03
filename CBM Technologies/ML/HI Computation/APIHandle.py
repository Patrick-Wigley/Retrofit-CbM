import thingspeak

import sys

from SECRETS import CHANNEL_ID, API_KEY

import os
import pandas as pd
from pandas.errors import EmptyDataError

FEATURES_PATH = "./saves/APIData/Features.csv"

import requests


def pull_features():
    if not os.path.isfile(FEATURES_PATH): 
        with open(FEATURES_PATH, "w") as f: f.close()
    
    try:
        locally_acquired_features = pd.read_csv(FEATURES_PATH)
        locally_acquired_features.index = locally_acquired_features["created_at"]
        locally_acquired_features.index = pd.to_datetime(locally_acquired_features.index, format="%Y-%m-%d %H:%M:%S") # ThingSpeak Format
        locally_acquired_features.drop(columns=["created_at"], inplace=True)
        
        # locally_acquired_features

    except EmptyDataError:
        print("The file is empty or has no columns to parse.")
        locally_acquired_features = pd.DataFrame()
    

    # API Handle (Get new rows)
    thingspeak_acquired_features = pd.DataFrame()

    URL = f"https://thingspeak.mathworks.com/channels/{CHANNEL_ID}/feeds.json"
    if locally_acquired_features.empty:
        response = requests.get(URL, params={"api_key": API_KEY})
    else:
        last_entry = locally_acquired_features.tail(n=1).index[-1] + pd.Timedelta(seconds=1) # Add a second so it doesnt take this last row again (only rows after this last row)
        last_entry = str(last_entry)
        print("Pulling data Stating from: ", last_entry)
             
        response = requests.get(URL, params={"api_key": API_KEY, "start":last_entry})

    data = response.json()
    meta:dict = data["channel"]
    feeds = data["feeds"]
    # get field names
    feature_fields = {}
    # 8 Fields
    for i in range(1, 8+1):
        if f"field{i}" in list(meta.keys()):
            feature_fields.update({f"field{i}": meta[f"field{i}"]})


    new_rows = pd.DataFrame(feeds)
    # new_rows["created_at"] = new_rows["created_at"].apply(lambda row: str(row).replace("T", " ").replace("Z", ""))
    
    if not new_rows.empty:
        new_rows.index = new_rows["created_at"]
        new_rows.index = pd.to_datetime(new_rows.index, format="%Y-%m-%dT%H:%M:%SZ") # ThingSpeak Format

        # Rename ThingSpeak fields (field1, field2, etc) to feature names (RMS, Kurtosis, etc)
        new_rows.rename(columns=feature_fields, inplace=True)
        new_rows.drop(columns=["created_at", "entry_id"], inplace=True)

    
    features = pd.concat([locally_acquired_features, new_rows])
    
    features.to_csv(FEATURES_PATH)
    
    return features

    # for record in feeds:
    #     print(record)
    #     new_row = {feature_fields[]}
    #     thingspeak_acquired_features.add

    # thingspeak


    # for datapoint in data["feeds"]:
    #     print(f"Time: {datapoint['created_at']}")
    #     # Access any field (field1, field2, etc.)
    #     print(f"Field 1: {datapoint.get('field1')}, Field 2: {datapoint.get('field2')}")



    # channel = thingspeak.Channel(id=CHANNEL_ID, api_key=API_KEY)
    # if locally_acquired_features.empty:
    #     print("No features pulled from API yet")
    #     # Pull all data
    #     channel.get_field()

    # else:
    #     pass # Will need to read last row in csv, get its timestamp and pull all data AFTER this timestamp.





    #channel.    

if __name__ == "__main__":
    pull_features()