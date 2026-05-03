import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler



features = [
    # Time Domain Features
    "rms",
    "std",
    "ptp",
    "kurtosis",
    "skew",
    "crest",

    # Frequency Domain Features
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_total",
    "spectral_entropy",
    "frequency_peak"
]

def feature_extraction(csv_delemiter="\t", dataset_in_use="./Datasets/NASABearing/1st_test/1st_test/") -> pd.DataFrame:
    df = pd.DataFrame(columns=features)


    DATAPOINTS_PER_FILE = 20480
    SAMPLE_RATE = 20000 # 20 kHz


    time_step = 1 / SAMPLE_RATE
    # np.arange returns arr of evenly spaced values within the given interval 
    # so a float timestamp that can be used in timedelta at the 
    # files starting point (such 2004.02.19.06.22.39) - there are 20480 
    # datapoints which need time indexing/stamping
    # Kind reminds me of pandas resample method but for relative (not actual) time in a given time period
    timestamps = np.arange(DATAPOINTS_PER_FILE) * time_step
    #print(timestamps)


    with open(dataset_in_use+os.listdir(dataset_in_use)[0]) as file:
        line = file.readline()
        NUMERICAL_CHARS = [".", "-", "+"] # Doesnt count unary values, decimal points as delimeters
        for char in line:
            if not char.isnumeric() and char not in NUMERICAL_CHARS:
                print(f"Suspected Delimiter: '{char}'")
                csv_delemiter = char
                break
        cols = line.count(csv_delemiter) + 1
        print(f"Bearings (coloums) found: {cols}")
        BEARINGS_COUNT = cols


    for file_reading_interval in os.listdir(dataset_in_use):
    # Each iteration is a burst (file contains one burst)


        # print(file_reading_interval)
        time_interval_df = pd.read_csv(
            dataset_in_use+file_reading_interval, 
            delimiter=csv_delemiter, 
            # [b1, b2, ..., bn]
            names=["b"+str(i+1) for i in range(BEARINGS_COUNT)], header=None)

        start = pd.to_datetime(file_reading_interval, format="%Y.%m.%d.%H.%M.%S")
        time_interval_df.index = start + pd.to_timedelta(timestamps, unit="s")

        # print(f"reading starts at: {time_interval_df.index[0]}")
        # print(f"reading ends at: {time_interval_df.index[-1]}")
        # print(time_interval_df)
        # break

        try:
            """Bearings Readings over this Signal"""
            bearings_readings = time_interval_df["b5"]
        except KeyError as err:
            print("\n[CAUGHT ERROR]: File(s) for set selected doesnt have this many columns?")
            break


        features_dict = feature_computation(bearings_readings, file_reading_interval, DATAPOINTS_PER_FILE)


        df.loc[len(df)] = features_dict

    df.index = os.listdir(dataset_in_use)
    df.index = pd.to_datetime(df.index, format="%Y.%m.%d.%H.%M.%S")

    return df



def feature_computation(bearings_readings, file_reading_interval, DATAPOINTS_PER_FILE):
    # TIME DOMAIN
    td_features_dict = {}

    # Numpy array
    x:list = np.asarray(bearings_readings)
    mu:float = np.mean(x)
    N:int = len(x)

    td_features_dict["id"] = file_reading_interval

    # Features Extraction 
    # Root Mean Square - Overall Energy in Signal
    RMS = np.sqrt(np.sum(x**2) / N)
    td_features_dict["rms"] = RMS # np.sqrt(np.mean(bearings_readings**2))

    # Standard Deviation - Spread/Var
    Std = np.sqrt(np.sum((x - mu)**2) / N)
    td_features_dict["std"] = Std # np.std(bearings_readings)


    # Peak to Peak - Distance from max and min
    Peak_max:float = np.max(x)
    Peak_min:float = np.min(x)
    PTP:float = Peak_max - Peak_min
    td_features_dict["ptp"] = PTP # np.ptp(bearings_readings)

    # Skewness - 3rd Central Moment - Symmetric difference (more prominant on negative or positive side)
    Sk = (np.sum((x-mu)**3)) / ((N-1)*Std**3)
    td_features_dict["skew"] = Sk # stats.skew(bearings_readings)


    # Kurtosis - 4th Central Moment - Determines flatness or peakiness (is the distribution more similar therefore flatter or does it contain some impulsive anomalies therefore peakier)
    Ku = (np.sum((x-mu)**4)) / ((N-1)*Std**4)
    td_features_dict["kurtosis"] = Ku #stats.kurtosis(bearings_readings, fisher=False)  # Fisher (if True) subtracts 3 from result (3 is standardised as normal distribution) so the normal distribution score is then 0.0

    # Crest Factor - Early warning sign for Impulsive Behaviour -  Indicates impact incuring within the bearing (contacts between the balls & the raceway track)
    # Takes max (Peak max) over the absolute values of x here: https://www.mdpi.com/2075-1702/5/4/21, the absolute x is not explicit here: https://www.researchgate.net/publication/286318685_Survey_of_condition_indicators_for_condition_monitoring_systems
    Peak_max_abs:float = np.max(np.abs(x))
    CF = Peak_max_abs / RMS
    td_features_dict["crest"] = CF





    # FREQUENCY DOMAIN
    fd_features_dict = {}
    # Convert Time-Domain to Frequency-Domain
    # FFT is the Fast Fourier Transform - It is the optimised Discrete Fourier Transform (DFT) 
    fft_b1_burst = np.fft.rfft(list(bearings_readings))

    # Magnitude Spectrum consists of Absolute (positives) where m_n >= 0
    M = np.abs(fft_b1_burst)

    frequencies = np.fft.rfftfreq(len(bearings_readings), d=1/DATAPOINTS_PER_FILE) # 1/sample rate per burst (20480)
    # First reading is normally super high (the first value when using the FFT 
    # formular is set to the sum of all the samples in this signal instance - apprently)
    # so set it to 0 to remove this dominance
    M[0] = 0

    # Index is n, n = 1, ... F Mathematically - n = 0, ..., F-1 Computationally
    F = len(frequencies)    

    MSum = np.sum(M) # Use to optimise beloew & bring some of them down to constant time-complexity O(1)

    # Bins in FFT are frequency slots (indicates strength of vibration at that frequency)
    # The readings (20480 rows) becoems: 20480 / 2 + 1 = 10241 frequency bins 


    # - spectral centroid (center of mass of spectrum)
    SC = np.sum(frequencies * M) / MSum
    fd_features_dict["spectral_centroid"] = SC

    # - Spectral bandwidth (spread of frequencies)
    SB = np.sqrt(np.sum(((frequencies - SC) ** 2) * M) / MSum)
    fd_features_dict["spectral_bandwidth"] = SB

    # - Spectral Energy (Total overall vibration energy)
    SE = np.sum(M**2)
    fd_features_dict["spectral_total"] = SE

    # NOTE - Using Spectal Entropy brought the triggered anomaly onset period back 2 days for the NASA DS, Set 1 (Earlier Warning!)
    # Spectral Entropy - Measure peakiness over the M spectrum
    # P_n is Probabiltity Distribution - Found in literature typically, & MathWorks examples as P(m) where m is the sample (this fft frequency's bins) :) - n is indicies over our fft samples 
    PSum = MSum 
    P_n = M / PSum

    # Prevent NAN issue - log(0 or -P_n) = undefined (NaN) - This occurs becaues the first value in P_n is 0 
    P_n_safe = P_n[P_n > 0]

    # TODO clarify this is correct
    SEN = -np.sum(P_n_safe * np.log2(P_n_safe)) / np.log2(F)
    fd_features_dict["spectral_entropy"] = SEN
    # Spectral entropy came down a tiny amount over the lifetime of the eventually broken bearing.


    # EXTRA (Not mentioned in methodology & not used here)

    # - Peak frequency (dominant vibration frequency)
    # In OpenAE, they do the rfftfreq conversion as part of this feature equation. We already compute rfft therefore, 
    # we can just do the following: 
    # np.argmax Finds & Returns the indices of the maximum values along an axis (idx of largest val in M).
    SPF = frequencies[np.argmax(M)] 
    fd_features_dict["frequency_peak"] = SPF


    # High Freqs energy ratio (threshold is >5kHz) 
    # - High-frequency energy ratio (Early fault indicator?)
    # THRESHOLD = 5000
    # fd_features_dict["high_frequency_ratio"] = np.sum(M[frequencies > THRESHOLD]**2) / fd_features_dict["spectral_total"] 
    ##########


    return {**td_features_dict, **fd_features_dict}


if __name__ == "__main__":
    DATASET_LOC = "./Datasets/"
    NASA_BEARING_DATASET_LOC = DATASET_LOC + "NASABearing/"
    DEBUG = True

    # For all sets_directories (not files) in dataset dir - Path.join is concatenates full path str lit to directory
    sets_directories = [os.path.join(NASA_BEARING_DATASET_LOC, d)+f"/{d}/" for d in  os.listdir(NASA_BEARING_DATASET_LOC) if os.path.isdir(os.path.join(NASA_BEARING_DATASET_LOC, d))]

    ## Select Dataset ##
    dataset_in_use = sets_directories[0]

    if DEBUG:
        print("Available sets in Dataset are:")
        for i, dir in enumerate(sets_directories):
            print(f"[{i}] {dir}")
        # Using set/run 2
        print(f"Using set: {dataset_in_use}")


    print(f"Count of bursts/files: {len(os.listdir(dataset_in_use))}")

    df = feature_extraction()
    print(df)



def feature_standardise(features_df:pd.DataFrame, features_names:list) -> pd.DataFrame:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df[features_names].values)

    return pd.DataFrame(features_scaled, columns=features_names, index=features_df.index)