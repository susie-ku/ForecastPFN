import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

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
        self.label_len = 18
#         self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]

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

        len_train = np.minimum(self.real_size, len(series_train))
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
            'n': target, 
        }
        return out
    

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1_.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 scaler=StandardScaler(), train_budget=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler
        self.train_budget = train_budget

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # print(cols)
        # print(self.target)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        train_start = 0
        if self.train_budget:
            train_start = max(train_start, num_train -
                              self.seq_len - self.train_budget)

        border1s = [train_start, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[0:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_stamp_original = df_raw[border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # ????
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        # # seq_x_original = self.data_stamp_original['date'].values[s_begin:s_end]
        # # seq_y_original = self.data_stamp_original['date'].values[r_begin:r_end]
        
        ### _process_tuple
        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        seq_x_mark = seq_x_mark.astype(np.float32)
        seq_y_mark = seq_y_mark.astype(np.float32)
        ###
        
        ### _ForecastPFN_process_batch
            ###_process_tuple
        if np.all(seq_x == seq_x[0]):
            seq_x[-1] += 1
        history = seq_x.copy()
        mean = history.mean()
        std = history.std()
        history = (history - mean) / std
        history_mean = np.nanmean(history[-6:])
        history_std = np.nanstd(history[-6:])
        local_scale = (history_mean + history_std + 1e-4)
        history = np.clip(history / local_scale, a_min=0, a_max=1)
             
        if seq_x.shape[0] != 100:
            seq_x_mark = seq_x_mark.astype(np.int64)
            if seq_x.shape[0] > 100:
                target = seq_x_mark[-100:, :]
                history = history[-100:, :]
            else:
                target = np.pad(seq_x_mark, (0, 100-seq_x.shape[0]))
                history = np.pad(history, (0, 100-seq_x.shape[0]))

            history = np.repeat(np.expand_dims(history, axis=0), self.pred_len, axis=0)[:, :, 0]
            ts = np.repeat(np.expand_dims(target, axis=0), self.pred_len, axis=0)

        else:
            ts = np.repeat(np.expand_dims(seq_x_mark, axis=0), self.pred_len, axis=1).astype(np.int64)
            history = history.astype(np.float32)
            
        task = np.full((self.pred_len, ), 1)  
        target_ts = np.expand_dims(seq_y_mark[-self.pred_len:, :], axis=1).astype(np.int64)
        model_input = {'ts': ts, 'history': history, 'target_ts': target_ts, 'task': task}

        return model_input, mean, std

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 scaler=StandardScaler()):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = scaler
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # test
        # seq_x = seq_x #.float()
        # seq_y = seq_y #.float()

        # seq_x_mark = seq_x_mark # .float()
        # seq_y_mark = seq_y_mark # .float()
        ###
        
        # _process_tuple
        # print(seq_x.shape)
        # print(seq_y.shape)
        # print(seq_x_mark.shape)
        # print(seq_y_mark.shape)
        # if torch.all(seq_x == seq_x[0]):
        #     seq_x[-1] += 1

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
