from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from distutils.util import strtobool
from keras import backend

# from data_provider.data_factory import data_provider
# from ..utils.metrics import smape

def smape(y_true, y_pred):
    """ Calculate Armstrong's original definition of sMAPE between `y_true` & `y_pred`.
        `loss = 200 * mean(abs((y_true - y_pred) / (y_true + y_pred), axis=-1)`
        Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        Returns:
        Symmetric mean absolute percentage error values. shape = `[batch_size, d0, ..
        dN-1]`.
        """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    diff = tf.abs(
        (y_true - y_pred) /
        backend.maximum(y_true + y_pred, backend.epsilon())
    )
    return 200.0 * backend.mean(diff, axis=-1)

def smapeofa(true, pred, agg=np.mean):
    smape = 200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)
    if agg is None:
        return smape
    else:
        return agg(smape)
    
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

class M4ZeroShotDataset(Dataset):   
    def __init__(self, train, horizon, history, test=None, max_size=500):
        self.make_test = False
        if test is None:
            self.make_test = True
        else:
            self.test = self.load_data(test)

        self.train = self.load_data(train)

        self.horizon = horizon
        self.history = history
        self.max_size = max_size


    def load_data(self, data):
        if isinstance(data, list):
            timeseries = data
        else:
            if isinstance(data, str):
                if data.split('.')[-1] == 'tsf':
                    data = convert_tsf_to_dataframe(data)[0]
                    timeseries = [self.dropna(ts).astype(np.float32) for ts in data.loc[:, 'series_value'].values]
                else:
                    data = pd.read_csv(data) 
                    timeseries = [self.dropna(ts).values.astype(np.float32) for n, ts in data.iloc[:, 1:].iterrows()]
        return timeseries


    @staticmethod
    def dropna(x):
        return x[~np.isnan(x)]

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):

        series_train = self.train[index]
        if self.make_test:
            series_test = series_train[-self.horizon:]
            series_train = series_train[:-self.horizon]
        else:
            series_test = self.test[index]

        len_train = np.minimum(self.history, len(series_train))
        pad_size = self.max_size - len_train

        x_train = series_train[-len_train:]
        x_val = series_test

        train_min = np.min(x_train)
        train_ptp = np.ptp(x_train)
        if np.isclose(train_ptp, 0):
            train_ptp = 1
        x_train = (x_train - train_min) / train_ptp
        x_val = (x_val - train_min) / train_ptp
        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)

        mask_train = np.pad(np.ones(len(x_train)), (0, pad_size), mode = 'constant')
        x_train = np.pad(x_train, (0, pad_size), mode = 'constant')
        
        
        out = {
            'series': x_train.astype('float32'),
            'target': x_val.astype('float32'),
            'mask_series': mask_train,
            'min': train_min,
            'ptp': train_ptp,
        }
        return out
    
def sliding_window_view(a, window):
    """Generate window view."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1,) + (window,)
    strides = a.strides[:-1] + (a.strides[-1],) + a.strides[-1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a, window, step=1, from_last=True):
    """from_last == True - will cut first step-1 elements"""
    sliding_window = (
        sliding_window_view(a, window)
        if np.__version__ < "1.20"
        else np.lib.stride_tricks.sliding_window_view(a, window)
    )
    return sliding_window[(len(a) - window) % step if from_last else 0 :][::step]


class TopInd:
    def __init__(
        self, n_target=7, history=100, step=3, from_last=True, test_last=True, date_col=None, scheme=None, **kwargs
    ):
        self.n_target = n_target
        self.history = history
        self.step = step
        self.from_last = from_last
        self.test_last = test_last
        self.date_col = date_col

    @staticmethod
    def _timedelta(x):
        delta = pd.to_datetime(x).diff().iloc[-1]
        if delta <= pd.Timedelta(days=1):
            return pd.to_datetime(x).diff().fillna(delta).values, delta

        if delta > pd.Timedelta(days=360):
            d = pd.to_datetime(x).dt.year.diff()
            delta = d.iloc[-1]
            return d.fillna(delta).values, delta
        elif delta > pd.Timedelta(days=27):
            d = pd.to_datetime(x).dt.month.diff() + 12 * pd.to_datetime(x).dt.year.diff()
            delta = d.iloc[-1]
            return d.fillna(delta).values, delta
        else:
            return pd.to_datetime(x).diff().fillna(delta).values, delta

    def read(self, data, plain_data=None):
        self.len_data = len(data)
        self.date_col = self.date_col
        self.time_delta = self._timedelta(data[self.date_col])[1]
        return self
        # TODO: add asserts

    def _create_test(self, data=None, plain_data=None):
        # for predicting future
        return rolling_window(
            np.arange(self.len_data if data is None else len(data)), self.history, self.step, self.from_last
        )[-1 if self.test_last else 0 :, :]

    def _create_data(self, data=None, plain_data=None):

        return rolling_window(
            np.arange(self.len_data if data is None else len(data))[: -self.n_target],
            self.history,
            self.step,
            self.from_last,
        )

    def _create_target(self, data=None, plain_data=None):
        return rolling_window(
            np.arange(self.len_data if data is None else len(data))[self.history :],
            self.n_target,
            self.step,
            self.from_last,
        )

    def _get_ids(self, data=None, plain_data=None, func=None, cond=None):
        date_col = pd.to_datetime(data[self.date_col])
        vals, time_delta = self._timedelta(data[self.date_col])
        ids = list(np.argwhere(vals != time_delta).flatten())
        prev = 0
        inds = []
        for split in ids + [len(date_col)]:
            segment = date_col.iloc[prev:split]
            if len(segment) > cond:
                ind = func(segment) + prev
                inds.append(ind)
            prev = split
        inds = np.vstack(inds)
        return inds

    def create_data(self, data=None, plain_data=None):
        return self._get_ids(data, plain_data, self._create_data, self.n_target + self.history)

    def create_test(self, data=None, plain_data=None):
        return self._get_ids(data, plain_data, self._create_test, self.history)

    def create_target(self, data=None, plain_data=None):
        return self._get_ids(data, plain_data, self._create_target, self.n_target + self.history)

class BenchZeroShotDataset(Dataset):   
    def __init__(self, horizon, history, real_size, train, borders, max_size=500, date_column = 'date', scale=True):

        self.horizon = horizon
        self.history = history
        self.max_size = max_size

        self.real_size = real_size
        self.borders = borders

        self.date_column = date_column
        self.scale = scale
        self.scaler = None
        self.targets = None

        train, val, test = self.load_data(train)

        L_split_data = val[date_column].values[(len(val) - history) if (len(val) - history) > 0 else 0]
        L_last_val_data = val[val[date_column] >= L_split_data]
        if (len(val) - history) < 0:
            L_split_data = train[date_column].values[(len(train) - (history- len(L_last_val_data))) if (len(train) - (history- len(L_last_val_data))) > 0 else 0]
            L_last_train_data = train[train[date_column] >= L_split_data]
            test_data_expanded = pd.concat((L_last_train_data, L_last_val_data, test))
        else:
            test_data_expanded = pd.concat((L_last_val_data, test))
        test_data_expanded = test_data_expanded.sort_values([date_column]).reset_index(
            drop=True
        )
        self.data = test_data_expanded
        slicer = TopInd(n_target=horizon, history=history, step=1, from_last=False, test_last=False, date_col=date_column)
        slicer.read(self.data)

        self.ids = slicer.create_data(self.data)
        self.ids_y = slicer.create_target(self.data)

        self.data = self.data.loc[:, self.targets].values
        
        # add time features
        # self.ts = self.data.loc[:, self.data_stamp].values
        # self.ts = self._ForecastPFN_time_features(self.data.loc[:, self.data_stamp].values)
        # self.data_stamp_original = df_raw[border1:border2]
        # test_data.data_stamp = self._ForecastPFN_time_features(list(test_data.data_stamp_original['date']))
        self.ts = self._ForecastPFN_time_features(list(test_data.data_stamp_original['date']))
        
    def _ForecastPFN_time_features(self, ts: np.ndarray):
        if type(ts[0]) == datetime.datetime:
            year = [x.year for x in ts]
            month = [x.month for x in ts]
            day = [x.day for x in ts]
            day_of_week = [x.weekday()+1 for x in ts]
            day_of_year = [x.timetuple().tm_yday for x in ts]
            return np.stack([year, month, day, day_of_week, day_of_year], axis=-1)
        ts = pd.to_datetime(ts)
        return np.stack([ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)


    def load_data(self, data):
        data = pd.read_csv(data)
        
        self.targets = list(data.columns.drop(self.date_column))

        if self.borders is None:
            TEST_SIZE = 0.2
            VAL_SIZE = 0.1

            train_val_split_data = data[self.date_column].values[
                int(data[self.date_column].nunique() * (1 - (VAL_SIZE + TEST_SIZE)))
            ]
            val_test_slit_data = data[self.date_column].values[
                int(data[self.date_column].nunique() * (1 - TEST_SIZE))
            ]

            train = data[data[self.date_column] <= train_val_split_data]
            val = data[
                (data[self.date_column] > train_val_split_data) & (data[self.date_column] <= val_test_slit_data)
            ]
            test = data[data[self.date_column] > val_test_slit_data]

        else:
            train = data.iloc[:self.borders[0]]
            val = data.iloc[self.borders[0]:self.borders[1]]
            test = data.iloc[self.borders[1]:self.borders[2]]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(train.loc[:, self.targets].values)
            train.loc[:, self.targets] = self.scaler.transform(train.loc[:, self.targets].values)
            val.loc[:, self.targets] = self.scaler.transform(val.loc[:, self.targets].values)
            test.loc[:, self.targets] = self.scaler.transform(test.loc[:, self.targets].values)
        

        return train, val, test

    def __len__(self):
        return int(len(self.ids) * len(self.targets))

    def __getitem__(self, index):
        #index = index + 1
        ind, target = index // len(self.targets), index % len(self.targets)
        series_train = self.data[self.ids[ind], target]
        series_test = self.data[self.ids_y[ind], target]
        
        seq_x_mark = self.ts[self.ids[ind]]
        seq_y_mark = self.ts[self.ids_y[ind]]

        len_train = np.minimum(self.real_size, len(series_train))
        pad_size = self.max_size - len_train

        x_train = series_train[-len_train:].astype(np.float32)
        x_val = series_test.astype(np.float32)
        seq_x_mark = seq_x_mark.astype(np.float32)
        seq_y_mark = seq_y_mark.astype(np.float32)
        
        if np.all(x_train == x_train[0]):
            x_train[-1] += 1
        history = x_train.copy()
        mean = np.nanmean(history)
        std = np.nanstd(history)
        history = (history - mean) / std
        history_mean = np.nanmean(history[-6:])
        history_std = np.nanstd(history[-6:])
        local_scale = (history_mean + history_std + 1e-4)
        history = np.clip(history / local_scale, a_min=0, a_max=1)
        
        if x_train.shape[0] != 100:
            seq_x_mark = seq_x_mark.astype(np.int64)
            if x_train.shape[0] > 100:
                target = seq_x_mark[-100:, :]
                history = history[-100:, :]
            else:
                target = np.pad(seq_x_mark, (0, 100-x_train.shape[0]))
                history = np.pad(history, (0, 100-x_train.shape[0]))

            history = np.repeat(np.expand_dims(history, axis=0), self.pred_len, axis=0)[:, :, 0]
            ts = np.repeat(np.expand_dims(target, axis=0), self.pred_len, axis=0)

        else:
            ts = np.repeat(np.expand_dims(seq_x_mark, axis=0), self.pred_len, axis=1).astype(np.int64)
            history = history.astype(np.float32)
            
        task = np.full((self.pred_len, ), 1)  
        target_ts = np.expand_dims(seq_y_mark[-self.pred_len:, :], axis=1).astype(np.int64)
        model_input = {'ts': ts, 'history': history, 'target_ts': target_ts, 'task': task}

        return model_input, mean, std
        
        
def test(model, loader, device):
    trues = []
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader):
            model_input = batch[0]
            mean = batch[1]
            std = batch[2]
            pred_vals = model(model_input).data.cpu()
            scaled_vals = pred_vals['result'].numpy().T.reshape(-1) * pred_vals['scale'].numpy().reshape(-1)
            scaled_vals = scaled_vals * std + mean
            true_vals = model_input['history']
            print(scaled_vals.shape)
            print(true_vals.shape)

            trues.append(list(scaled_vals.numpy()))
            preds.append(list(true_vals.numpy()))
              
    trues = np.vstack(trues)
    preds = np.vstack(preds)
    return preds, trues

if __name__ == '__main__':
    SEQ_LEN = 250
    PATH = '/home/jovyan/kkuvshinova/ForecastPFN_accelerate/academic_data'
        # configs = { 
        #     'Weather_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/weather/weather.csv', 'borders': None, 'max_size':500}, 
        #     #'Traffic_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/traffic/traffic.csv', 'borders': None, 'max_size':500}, 
        #     #'Electricity_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/electricity/electricity.csv', 'borders': None, 'max_size':500}, 
        #     'ILI_24_148': {'horizon': 24, 'history': 148, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}, 
        #     'ETTh1_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh1.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
        #     'ETTh2_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh2.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
        #     'ETTm1_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm1.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
        #     'ETTm2_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm2.csv', 'borders': [34560, 46080, 57600], 'max_size':500},
        #     'Demand': {'horizon': 30, 'history': 50, 'real_size': SEQ_LEN, 'train': '/media/ssd-3t/Kostromina/pyboost_experiments/data/demand/demand_scaled_transformer_format.csv', 'borders': None, 'max_size':500, 'scale': False}, 
        # }
    config = {'horizon': 24, 'history': 148, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}
    device = 'cuda'
    dataset = BenchZeroShotDataset(**config)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    pretrained = tf.keras.models.load_model("../saved_weights/", custom_objects={'smape': smape})
    test(pretrained, loader, device)