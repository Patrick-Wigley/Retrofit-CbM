import pandas as pd
import os

RECORDINGS_DIR = "./Recordings/"
TESTING = RECORDINGS_DIR + "/Testing/"

DIR_IN_USE = TESTING
vib_folder:list = os.listdir(DIR_IN_USE)

FILE_SAMPLES:int = 1 # len(vib_fold)

df:pd.DataFrame
for vib_file in vib_folder:  
    df = pd.read_csv(DIR_IN_USE + f"/{vib_file}", names=["AX", "AY", "AZ"])

print(df)
df1 = df.diff()
print(df1)





# map(lambda vib_file: print(vib_file), vib_folder.)


# print(vib_folder.__next__())
# with open("./Recordings/VR")