import pandas as pd
import numpy as np
import glob
import os

path = '/media/ssd-3t/kkuvshinova/time_series/ForecastPFN/academic_data'
mystep = 0.4

for root, dirs, _ in os.walk(path):
    for d in dirs:
        path_sub = os.path.join(root, d) # this is the current subfolder
        for filename in glob.glob(os.path.join(path_sub, '*.csv')):
            # print(filename)
            df = pd.read_csv(filename)
            df.columns = df.columns.str.replace(' ', '_')
            # print(" ".join(df.columns.drop(['date']).tolist()))
            with open(f"{filename}.txt".replace(".csv", ""), "w+") as text_file:
                text_file.write(" ".join(df.columns.drop(['date']).tolist()))
            # df.to_csv('_' + filename, index=False)
            df.to_csv(filename.replace(".csv", "") + '_.csv')