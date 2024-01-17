config = {}
maps = {'Weather': 24, 'Traffic': 168, 'Electricity': 24, 'ILI': 52, 'ETTh1': 12, 'ETTh2': 12, 'ETTm1': 48, 'ETTm2': 48}
for n_h, horizon in enumerate([96, 192, 336, 720]):
    for history_size_coef in [1, 2, 3, 4, 5, 6, 7, 8]: 
        for name, period in maps.items():
            config['horizon'] = horizon if name != 'ILI' else int(12 * (n_h + 2))
            config['real_size'] = int(maps[name] * (2 ** history_size_coef)) if name != 'ILI' else int(maps[name] * history_size_coef)
            if config['real_size'] >= 500:
                config['real_size'] = 490
            with open(f"path{name}_history.txt", "w+") as text_file:
                text_file.write(" ".join(df.columns.drop(['date']).tolist()))
                
                
path = '/media/ssd-3t/kkuvshinova/time_series/ForecastPFN/academic_data'
mystep = 0.4

for root, dirs, _ in os.walk(path):
    for d in dirs:
        path_sub = os.path.join(root, d) # this is the current subfolder
        for filename in glob.glob(os.path.join(path_sub, '*.csv')):