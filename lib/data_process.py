# coding = utf-8
"""
作者   : Hilbert
时间   :2021/4/11 16:12
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
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.his = torch.stack(transposed_data[0], 0)
        self.q = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.his = self.his.pin_memory()
        self.q = self.q.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


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


