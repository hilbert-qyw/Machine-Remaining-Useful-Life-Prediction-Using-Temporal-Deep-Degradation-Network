# coding = utf-8
"""
作者   : Hilbert
时间   :2022/6/29 9:15
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
import torchvision
import torchvision.transforms as transforms
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


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

if (file_number == 1) or (file_number == 3):
    sensor_num = 16
else:
    sensor_num = 24


outputdir = fr'../turbofan_{model_name}_result/' + f"{file_name}/ce{max_life}_{min_max_norm}_times{times}"
mkdir(outputdir)


def load_data():
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

    file_name = 'FD00' + str(file_number)
    # FD001 and FD003 feature select
    if (file_number == 1) or (file_number == 3):
        column = [4, 5, 9, 10, 14, 20, 22, 23]
        training_data, testing_data = feature_selection(training_data, column), \
                                      feature_selection(testing_data, column)

    # label normalization
    training_data[:, -1] /= max_life
    testing_data[:, -1] /= max_life
    sensor_num = training_data.shape[1] - 3

    return training_data, testing_data, sensor_num



def train_TDRL(config, checkpoint_dir=None):
    training_data, testing_data, sensor_num = load_data()
    training_data, val_data = train_split(training_data, ptrain=config['ptrain'], seed=config['seed'])

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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=False,
                            shuffle=False)

    # testing data
    test_input = GenerateInputSample(data=testing_data, cycle_step=cycle_step, sensor_num=sensor_num, stride=stride)
    x_test, y_test = test_input.testing_data()
    test_loader = []
    for (x_sample, y_sample) in zip(x_test, y_test):
        sample_dataset = TensorDataset(torch.Tensor(x_sample), torch.Tensor(y_sample))
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, collate_fn=collate_wrapper,
                                   pin_memory=False, shuffle=False)
        test_loader.append(sample_loader)

    net = MCNNsAttention(ts_shape=(sensor_num, cycle_step), pool_factor=config['pool_factor'],
                         kernel_size=config['kernel_size'], moving_average=config['moving_average'],
                         dim_atten=config['dim_atten'], dim_reg=config['dim_reg'],drop_r=config['drop_r'])

    net = net.double()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    loss = nn.MSELoss()

    if checkpoint_dir:
        # 模型的状态、优化器的状态
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)



    trainer = Trainer(model=net, criterion=loss, train_dataloader=train_loader, verbose=True, maxlife=max_life,
                      saving_path=outputdir, val_dataloader=val_loader, test_dataloader=test_loader, many2one=True)
    trainer.train(epochs=epochs, optimizer=optimizer)


def main(num_samples=20, max_num_epochs=10, gpus_per_trial=2):
    # 全局文件路径
    # 加载训练数据
    load_data()
    # 配置超参数搜索空间
    # 每次实验，Ray Tune会随机采样超参数组合，并行训练模型，找到最优参数组合
    config = {
        # 自定义采样方法
        'pool_factor': tune.choice([2, 3, 4]),
        'kernel_size': tune.choice([2, 3, 5]),
        'moving_average': [tune.randint(2, 5), tune.randint(5, 8), tune.randint(8, 10)],
        'dim_atten': tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
        'dim_reg': tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        'drop_r': tune.quniform(0.2, 0.6, 0.1),
        'lr': tune.loguniform(1e-6, 1e-3),
        'batch_size': tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        'p_train': tune.choice([0.8, 0.85, 0.9]),
        'seed': tune.uniform(1, 10)
    }

    # ASHAScheduler会根据指定标准提前中止坏实验
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=3)

    # 在命令行打印实验报告
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    # 执行训练过程
    result = tune.run(
        partial(train_TDRL),
        # 指定训练资源
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    # 找出最佳实验
    best_trial = result.get_best_trial("loss", "min", "last")
    # 打印最佳实验的参数配置
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # 打印最优超参数组合对应的模型在测试集上的性能
    best_trained_model = MCNNsAttention(ts_shape=(sensor_num, cycle_step),
                                        pool_factor=best_trial.config['pool_factor'],
                                        kernel_size=best_trial.config['kernel_size'],
                                        moving_average=best_trial.config['moving_average'],
                                        dim_atten=best_trial.config['dim_atten'],
                                        dim_reg=best_trial.config['dim_reg'],
                                        drop_r=best_trial.config['drop_r'])

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)

