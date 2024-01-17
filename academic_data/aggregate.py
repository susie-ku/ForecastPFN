import pandas as pd

<<<<<<< HEAD
path = 'traffic/traffic.csv'
out_path = path.replace('.csv', '_agg_mean.csv')
=======
path = '/home/chira/ForecastPFN/academic_data/open_src_datasets/room_temp.csv'
out_path = path.replace('.csv', '_agg.csv')
>>>>>>> 59f56d7 (new datasets)

df = pd.read_csv(path)
df = df.drop(['id'], axis=1)
df['date'] = pd.to_datetime(df['date'])
df.groupby([df['date'].dt.date]).mean().to_csv(out_path)
