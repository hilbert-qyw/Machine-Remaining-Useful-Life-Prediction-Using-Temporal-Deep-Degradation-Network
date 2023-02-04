# coding = utf-8
"""
作者   : Hilbert
时间   :2021/10/16 9:52
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
import pandas as pd
import matplotlib.pyplot as plt
import torch
from lib.util import *
from lib.raw_data_processing import get_CMAPSSData
from lib.temporal_deep_remaining_life import *
from torch import optim
import argparse

# example python TDRL_model_training.py --file_number 1 --max_life 1 --time 1 --norm symmetric
# parser = argparse.ArgumentParser(description="input hyper-parameters")
# parser.add_argument("--file_number", help="the order of file", required=True)
# parser.add_argument("--max_life", help="normalization", required=True)
# parser.add_argument("--times", help="number of training", required=True)
# parser.add_argument("--norm", help="Type of parameter normalization", required=True, type=str)
# args = parser.parse_args()

"""
hyper-parameters:
cycle_step: the length of time window
sensor_num: the number of sensors
file_number: sub-dataset
max_life: when max_life=1 means true value
padding_zero: padding first cycle or zero
stride: window movement length
label_norm: label normalization
"""
cycle_step = 64
batch_size = 16
epochs = 50
lr = 1e-6
stride = 1
model_name = 'MCNNsAttention'
# file_number = int(args.file_number)
# max_life = int(args.max_life)
# times = int(args.times)
# min_max_norm = args.norm

# direct assignment
file_number = 1
max_life = 125
times = 1
min_max_norm = 'symmetric'

# file path
file_name = 'FD00' + str(file_number)

# generate processed data
# CMAPSS = get_CMAPSSData(
#     file_path='../data/CMAPSSData/raw_data/',
#     outputdir='../data/CMAPSSData/processed_data/',
#     save_training_data=True,
#     save_testing_data=True,
#     files=[1, 2, 3, 4],
#     MAXLIFE=max_life,
#     min_max_norm=min_max_norm
# )

read_CMAPSS = get_CMAPSSData(
    file_path='../data/CMAPSSData/raw_data/',
    outputdir='../data/CMAPSSData/processed_data/',
    save_training_data=False,
    save_testing_data=False,
    files=[file_number],
    MAXLIFE=125,
    min_max_norm=min_max_norm
)

training_data, testing_data, _, _ = read_CMAPSS.read_processed_data()


# FD001 and FD003 feature select
if (file_number == 1) or (file_number == 3):
    column = [4, 5, 9, 10, 14, 20, 22, 23]
    training_data, testing_data = feature_selection(training_data, column), \
                                  feature_selection(testing_data, column)

# label normalization
training_data[:, -1] /= max_life
testing_data[:, -1] /= max_life

sensor_num = training_data.shape[1] - 3
outputdir = fr'../turbofan_{model_name}_result/' + f"{file_name}/ce{max_life}_{min_max_norm}_times{times}"
mkdir(outputdir)

# generate input dataycle{cycle_step}_sensor{sensor_num}_batchsize{batch_size}" \
#                                    f"_epochs{epochs}_lr{lr}_maxlif
training_data, val_data = train_split(training_data, ptrain=0.9)

#
# training_data = training_data[:100, :]

# training data
train_input = GenerateInputSample(data=training_data, cycle_step=cycle_step, sensor_num=sensor_num, stride=stride)
x_train, y_train = train_input.training_data()
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_wrapper,
                          pin_memory=False, shuffle=True)

# validation data
val_input = GenerateInputSample(data=val_data, cycle_step=cycle_step, sensor_num=sensor_num, stride=stride)
x_val, y_val = val_input.training_data()
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=False, shuffle=False)

# testing data
test_input = GenerateInputSample(data=testing_data, cycle_step=cycle_step, sensor_num=sensor_num, stride=stride)
x_test, y_test = test_input.testing_data()
test_loader = []
for (x_sample, y_sample) in zip(x_test, y_test):
    sample_dataset = TensorDataset(torch.Tensor(x_sample), torch.Tensor(y_sample))
    sample_loader = DataLoader(sample_dataset, batch_size=batch_size, collate_fn=collate_wrapper,
                               pin_memory=False, shuffle=False)
    test_loader.append(sample_loader)

transformations = {
    'identity': {
        'class': Identity,
        'params': []
    },
    'movingAvg': {
        'class': MovingAvg,
        'params': [2, 4, 8]       # window sizes
    },
    'downsample': {
        'class': Downsample,
        'params': [2]       # sampling rates
    }
}

# model
net = MCNNsAttention(ts_shape=(sensor_num, cycle_step), pool_factor=4, kernel_size=3,
                     transformations=transformations)


# net = TDRL(sensor_num, cycle_step, many2one=True)
net = net.double()
optimizer = optim.Adam(net.parameters(), lr=lr)
net.train()
loss = nn.MSELoss()

trainer = Trainer(model=net, criterion=loss, train_dataloader=train_loader, verbose=True, maxlife=max_life,
                  saving_path=outputdir, val_dataloader=val_loader, test_dataloader=test_loader, many2one=True)
trainer.train(epochs=epochs, optimizer=optimizer)
trainer.test(mode='best')