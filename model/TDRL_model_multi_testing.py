# coding = utf-8
"""
作者   : Hilbert
时间   :2022/4/8 19:29
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
cycle_step = 15
batch_size = 16
epochs = 50
lr = 0.0001
file_number = 4
max_life = 125
stride = 1
times = 1
min_max_norm = 'symmetric'
best_model_iter = 11
model_name = 'LSTM'
many2one = False
padding = False

# file path
file_name = 'FD00' + str(file_number)
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
outputdir = fr'../turbofan_{model_name}_result/' + f"{file_name}/cycle{cycle_step}_sensor{sensor_num}_batchsize{batch_size}" \
                                    f"_epochs{epochs}_lr{lr}_maxlife{max_life}_{min_max_norm}_times{times}"
mkdir(outputdir)

# testing data
test_input = GenerateInputMultiSample(data=testing_data, cycle_step=cycle_step, sensor_num=sensor_num, stride=1,
                                      many2one=many2one, padding=padding, padding_zero=False)
x_test, y_test = test_input.testing_data()
test_loader = []
for (x_sample, y_sample) in zip(x_test, y_test):
    sample_dataset = TensorDataset(torch.Tensor(x_sample), torch.Tensor(y_sample))
    sample_loader = DataLoader(sample_dataset, batch_size=len(sample_dataset), collate_fn=collate_wrapper,
                               pin_memory=False, shuffle=False)
    test_loader.append(sample_loader)

net = LSTM(sensor_num, cycle_step, many2one=many2one)
# net.load_state_dict(torch.load(outputdir+f'/best_model_{best_model_iter}.pt'))
net.load_state_dict(torch.load(outputdir+f'/last_model.pt'))
net.double()

# testing
Tester_testing_data = TesterMulti(model=net, test_dataloader=test_loader, saving_path=outputdir, maxlife=max_life,
                                  many2one=False)
Tester_testing_data.test()
Tester_testing_data.rul_curve(state='testing', number=34, saving_data=True)
# Tester_testing_data.tsne_result_of_sensor()

# # training
# Tester_training_data = Tester(model=net, test_dataloader=train_loader,
#                               saving_path=outputdir, maxlife=max_life, predict=True)
# Tester_training_data.sensor_data_curve(number=0)
# Tester_training_data.tsne_result_of_sensor(state='training', number=None)
# Tester_training_data.hidden_feature_curve(state='training', number=0)
# Tester_training_data.abstract_feature_curve(state='training', number=0)
# Tester_training_data.visualization_of_attention(number=0)
# Tester_training_data.degradation_pattern(number=60)
#
# # raw sensor
# Tester_training_data = Tester(model=net, test_dataloader=train_loader,
#                               saving_path=outputdir, maxlife=max_life, predict=False)
# Tester_training_data.sensor_data_curve(number=16)
#
#
# # CNN_layer effect
# CNN_layer_effect()
#
# # window size effect
# time_window_effect(rmse=None, dataset='FD001')

# performance metric
# performance_metric()