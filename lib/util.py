# coding = utf-8
"""
作者   : Hilbert
时间   :2021/4/11 15:53
"""
import sys
import os
from warnings import simplefilter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]  # 上一级目录
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch
from lib.temporal_deep_remaining_life import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def mkdir(path):
    """
    mkdir of the path
    :param input: string of the path
    return: boolean
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' is created!')
        return True
    else:
        print(path+' already exists!')
        return False


def read_file(file_number, txt_file_folder, csv_file_folder, processed_file_folder, state):
    file_name = state + '_FD00' + str(int(file_number))
    txt_file_path = txt_file_folder + file_name + r'.txt'
    csv_file_path = csv_file_folder + file_name + r'/' + file_name + r'.csv'
    processed_file_path = processed_file_folder + r'/normalized_' + state + r'_data_120_{}.npy'.format(file_number)
    txt_data = pd.read_table(txt_file_path, header=None, delim_whitespace=True)
    csv_data = pd.read_csv(csv_file_path, index_col=[0])
    processed_data = np.load(processed_file_path)
    # 按顺序排列
    processed_data = processed_data[np.lexsort(processed_data[:, ::-1].T)]
    return txt_data, csv_data, processed_data


def feature_selection(data, column):
    """
    sensor 1,5,6,10,16,18 and 19 in subset FD001 and FD003 exhibit constant sensor measurements
    delete the specified column
    :param data: input data
    :param column: column to be deleted
    """
    max_column = data.shape[1]
    if max(column) > max_column or min(column) < 0:
        print("The specified column is out of range!")
        exit()
    else:
        data = np.delete(data, column, axis=1)
    return data


def train_split(data, ptrain=0.75, seed=1):
    """
    engines in each subset are randomly selected as validation set
    :param data: raw_data
    :param ptrain: validation set ratio
    :return: training data and validation data
    """
    idx = list(range(len(np.unique(data[:, 0]))))
    random.seed(seed)
    random.shuffle(idx)
    train_idx = idx[: int(len(idx)*ptrain)]
    val_idx = idx[int(len(idx)*ptrain):]
    train_data = []
    for i in train_idx:
        engine_data = data[np.where(data[:, 0]==i)]
        train_data.append(engine_data)
    train_data = np.concatenate([train_data[i] for i in range(len(train_data))])
    val_data = []
    for j in val_idx:
        engine_data = data[np.where(data[:, 0]==j)]
        val_data.append(engine_data)
    val_data = np.concatenate([val_data[j] for j in range(len(val_data))])
    return train_data, val_data


class GenerateInputSample(object):
    def __init__(self, data, cycle_step, sensor_num, stride=1):
        self.data = data
        self.cycle_step = cycle_step
        self.sensor_num = sensor_num
        self.stride = stride

    # divide the data by id
    def sample_id(self):
        RUL = []
        sensors = []
        train_engine_id = np.unique(self.data[:, 0])
        for idx in train_engine_id:
            engine_data = self.data[np.where(self.data[:, 0] == idx)]
            RUL.append(engine_data[:, -1])
            sensors.append(engine_data[:, 2:-1])
        return sensors, RUL

    # sliding time window
    def sample_slice_and_cut(self, X, Y):
        out_X = []
        out_Y = []
        n_sample = len(X)
        for i in range(n_sample):
            tmp_ts = X[i]
            tmp_Y = Y[i]
            tmp_x = []
            tmp_y = []
            ts_padding = np.tile(tmp_ts[0], (self.cycle_step - self.stride, 1))
            tmp_ts = np.concatenate((ts_padding, tmp_ts), axis=0)
            for j in range(0, len(tmp_ts) - self.cycle_step + 1, self.stride):
                tmp_x.append(tmp_ts[j:j + self.cycle_step])
                tmp_y.append(tmp_Y[j])
            out_X.append(np.array(tmp_x))
            out_Y.append(np.array(tmp_y))

        return out_X, out_Y

    def training_data(self):
        # return training data: tensor format
        sensors, RUL = self.sample_id()
        x_train_set, y_train_set = self.sample_slice_and_cut(X=sensors, Y=RUL)
        x_train_set = torch.tensor(np.concatenate([x_train_set[i] for i in range(len(x_train_set))]))
        y_train_set = torch.tensor(np.concatenate([y_train_set[i] for i in range(len(y_train_set))]))
        return x_train_set, y_train_set

    def testing_data(self):
        # return testing data: tensor format
        sensors, RUL = self.sample_id()
        x_train_set, y_train_set = self.sample_slice_and_cut(X=sensors, Y=RUL)
        return x_train_set, y_train_set


class GenerateInputMultiSample(object):
    def __init__(self, data, cycle_step, sensor_num, stride=1, many2one=False, padding=False, padding_zero=True):
        self.data = data
        self.cycle_step = cycle_step
        self.sensor_num = sensor_num
        self.stride = stride
        self.many2one = many2one
        self.padding = padding
        self.padding_zero = padding_zero

    # divide the data by id
    def sample_id(self):
        RUL = []
        sensors = []
        train_engine_id = np.unique(self.data[:, 0])
        for idx in train_engine_id:
            engine_data = self.data[np.where(self.data[:, 0] == idx)]
            RUL.append(engine_data[:, -1])
            sensors.append(engine_data[:, 2:-1])
        return sensors, RUL

    # sliding time window
    def sample_slice_and_cut(self, X, Y):
        out_X = []
        out_Y = []
        n_sample = len(X)
        for i in range(n_sample):
            tmp_ts = X[i]
            tmp_Y = Y[i]
            tmp_x = []
            tmp_y = []
            if self.padding:
                if self.padding_zero:
                    ts_padding = np.zeros((self.cycle_step - self.stride, tmp_ts.shape[1]))
                else:
                    ts_padding = np.tile(tmp_ts[0], (self.cycle_step - self.stride, 1))
                    tmp_ts = np.concatenate((ts_padding, tmp_ts), axis=0)
                Y_padding = np.zeros((self.cycle_step - self.stride))
                tmp_Y = np.concatenate((tmp_Y, Y_padding))
                for j in range(0, len(tmp_ts) - self.cycle_step + 1, self.stride):
                    tmp_x.append(tmp_ts[j:j + self.cycle_step])
                    if self.many2one:
                        tmp_y.append(tmp_Y[j])
                    else:
                        tmp_y.append(tmp_Y[j:j + self.cycle_step])
                out_X.append(np.array(tmp_x))
                out_Y.append(np.array(tmp_y))
            else:
                for j in range(0, len(tmp_ts), self.stride):
                    if j + self.cycle_step < len(tmp_ts):
                        tmp_x.append(tmp_ts[j:j + self.cycle_step])
                        if self.many2one:
                            tmp_y.append(tmp_Y[j + self.cycle_step - 1])
                        else:
                            tmp_y.append(tmp_Y[j:j + self.cycle_step])
                    else:
                        tmp_x.append(tmp_ts[len(tmp_ts)-self.cycle_step:len(tmp_ts)])
                        if self.many2one:
                            tmp_y.append(tmp_Y[len(tmp_ts) - 1])
                        else:
                            if len(tmp_ts)-j-self.cycle_step == 0:
                                tmp_y.append(tmp_Y[len(tmp_ts)-self.cycle_step:len(tmp_ts)])
                            else:
                                tmp_y.append(np.concatenate([np.zeros((self.cycle_step+j-len(tmp_ts))),
                                                            tmp_Y[j:len(tmp_ts)]]))
                        out_X.append(np.array(tmp_x))
                        out_Y.append(np.array(tmp_y))
                        break
        return out_X, out_Y

    def training_data(self):
        # return training data: tensor format
        sensors, RUL = self.sample_id()
        x_train_set, y_train_set = self.sample_slice_and_cut(X=sensors, Y=RUL)
        x_train_set = torch.tensor(np.concatenate([x_train_set[i] for i in range(len(x_train_set))]))
        y_train_set = torch.tensor(np.concatenate([y_train_set[i] for i in range(len(y_train_set))]))
        return x_train_set, y_train_set

    def testing_data(self):
        # return testing data: tensor format
        sensors, RUL = self.sample_id()
        x_train_set, y_train_set = self.sample_slice_and_cut(X=sensors, Y=RUL)
        return x_train_set, y_train_set


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.sensor = torch.stack(transposed_data[0], 0)
        self.rul = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.sensor = self.sensor.pin_memory()
        self.rul = self.rul.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)