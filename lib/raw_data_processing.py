# coding = utf-8
"""
作者   : Hilbert
时间   :2021/10/14 22:16
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


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from lib.util import mkdir


class get_CMAPSSData:
    def __init__(self, file_path=None, outputdir=None, save_training_data=True,
                   save_testing_data=True, files=[1, 2, 3, 4], MAXLIFE=120, min_max_norm='z_score'):
        '''
        :param save: switch to load the already preprocessed data or begin preprocessing of raw data
        :param file_path: raw data saving path
        :param outputdir: processed data output path
        :param save_training_data: same functionality as 'save' but for training data only
        :param save_testing_data: same functionality as 'save' but for testing data only
        :param files: to indicate which sub dataset needed to be loaded for operations
        :param min_max_norm: switch to enable min-max normalization
                             z_score---> zero normalization  symmetric---> [-1, 1]   asymmetric--->[0, -1]
        '''
        self.FilePath = file_path
        self.OutputDir = outputdir
        self.SaveTrainingData = save_training_data
        self.SaveTestingData = save_testing_data
        self.files = files
        self.maxlife = MAXLIFE
        self.MinMaxNorm = min_max_norm
        self.ColumnName = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                           's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                           's15', 's16', 's17', 's18', 's19', 's20', 's21']
        mkdir(outputdir)

        for i in files:
            data_file = 'FD00' + str(i)

            if self.SaveTrainingData:  ### Training ###
                training_data_file = 'train_' + data_file
                self.read_raw_training_data(training_data_file)

            if self.SaveTestingData:
                testing_data_file = 'test_' + data_file
                rul_file = 'RUL_' + data_file
                self.read_raw_testing_data(testing_data_file, rul_file)

    def read_raw_training_data(self, training_data_file_name):
        """
        Read txt format file
        :param training_data_file_name: load subdataset name
        :return: save processed data
        """
        raw_training_data = pd.read_table(self.FilePath + f"{training_data_file_name}.txt", header=None, delim_whitespace=True)
        raw_training_data.columns = self.ColumnName

        self.norm_para = []

        if self.MinMaxNorm == 'z_score':
            #### standard normalization ####
            mean = raw_training_data.iloc[:, 2:len(raw_training_data)].mean()
            std = raw_training_data.iloc[:, 2:len(raw_training_data)].std()
            for idx, value in enumerate(std):
                if value < 1e-10:
                    std.iloc[idx] = 0
            std.replace(0, 1, inplace=True)
            self.norm_para.append((mean, std))
            raw_training_data.iloc[:, 2:len(raw_training_data)] = (raw_training_data.iloc[:, 2:len(raw_training_data)] - mean) / std
        else:
            max, min = raw_training_data.iloc[:, 2:len(raw_training_data)].max(), raw_training_data.iloc[:, 2:len(raw_training_data)].min()
            self.norm_para.append((max, min))
            if self.MinMaxNorm == 'asymmetric':
                raw_training_data.iloc[:, 2:len(raw_training_data)] = (raw_training_data.iloc[:, 2:len(raw_training_data)] - min) / (max - min)
            else:
                raw_training_data.iloc[:, 2:len(raw_training_data)] = (raw_training_data.iloc[:, 2:len(raw_training_data)] - min) / (max - min) * 2 - 1
        raw_training_data.fillna(0, inplace=True)
        raw_training_data['RUL'] = self.compute_rul_of_one_file(raw_training_data)
        print("finishing processed:\t " + training_data_file_name)

        train_values = raw_training_data.values
        np.save(self.OutputDir + f"{self.MinMaxNorm}_{training_data_file_name}_{self.maxlife}.npy", train_values)
        raw_training_data.to_csv(self.OutputDir + f"{self.MinMaxNorm}_{training_data_file_name}_{self.maxlife}.csv")


    def read_raw_testing_data(self, testing_data_file_name, RUL_name):
        raw_testing_data = pd.read_table(self.FilePath + f"{testing_data_file_name}.txt", header=None, delim_whitespace=True)
        RUL_data = pd.read_table(self.FilePath + f"{RUL_name}.txt", header=None, delim_whitespace=True)
        raw_testing_data.columns = self.ColumnName
        RUL_data.columns = ['RUL']

        if self.MinMaxNorm == 'z_score':
            #### standard normalization ####
            raw_testing_data.iloc[:, 2:len(raw_testing_data)] = (raw_testing_data.iloc[:, 2:len(raw_testing_data)]
                                                                 - self.norm_para[0][0]) / self.norm_para[0][1]
        else:
            if self.MinMaxNorm == 'asymmetric':
                raw_testing_data.iloc[:, 2:len(raw_testing_data)] = \
                    (raw_testing_data.iloc[:, 2:len(raw_testing_data)] - self.norm_para[0][1]) \
                    / (self.norm_para[0][0] - self.norm_para[0][1])
            else:
                raw_testing_data.iloc[:, 2:len(raw_testing_data)] = \
                    (raw_testing_data.iloc[:, 2:len(raw_testing_data)]- self.norm_para[0][1])\
                    / (self.norm_para[0][0] - self.norm_para[0][1]) * 2 - 1
        raw_testing_data.fillna(0, inplace=True)

        raw_testing_data['RUL'] = self.compute_rul_of_one_file(raw_testing_data, RUL_FD00X=RUL_data)
        print("finishing processed: \t" + testing_data_file_name)

        train_values = raw_testing_data.values
        np.save(self.OutputDir + f"{self.MinMaxNorm}_{testing_data_file_name}_{self.maxlife}.npy", train_values)
        raw_testing_data.to_csv(self.OutputDir + f"{self.MinMaxNorm}_{testing_data_file_name}_{self.maxlife}.csv")


    def read_processed_data(self):
        """
        :data_file:
        :return: processed data
        """
        for i in self.files:
            data_file = 'FD00' + str(i)
        return np.load(self.OutputDir + f"{self.MinMaxNorm}_train_{data_file}_{self.maxlife}.npy"), \
               np.load(self.OutputDir + f"{self.MinMaxNorm}_test_{data_file}_{self.maxlife}.npy"), \
               pd.read_csv(self.OutputDir + f'{self.MinMaxNorm}_train_{data_file}_{self.maxlife}.csv', index_col=[0]), \
               pd.read_csv(self.OutputDir + f'{self.MinMaxNorm}_test_{data_file}_{self.maxlife}.csv', index_col=[0])


    def compute_rul_of_one_file(self, FD00X, id='engine_id', RUL_FD00X=None):
        '''
        Input train_FD001, output a list
        '''
        rul = []
        # In the loop train, each id value of the 'engine_id' column
        if RUL_FD00X is None:
            for _id in set(FD00X[id]):
                rul.extend(self.compute_rul_of_one_id(FD00X[FD00X[id] == _id]))
            return rul
        else:
            rul = []
            for _id in set(FD00X[id]):
                rul.extend(self.compute_rul_of_one_id(FD00X[FD00X[id] == _id], int(RUL_FD00X.iloc[_id - 1])))
            return rul

    def compute_rul_of_one_id(self, FD00X_of_one_id, max_cycle_rul=None):
        '''
        Enter the data of an engine_id of train_FD001 and output the corresponding RUL (remaining life) of these data.
        type is list
        '''

        cycle_list = FD00X_of_one_id['cycle'].tolist()
        if max_cycle_rul is None:
            max_cycle = max(cycle_list)  # Failure cycle
        else:
            max_cycle = max(cycle_list) + max_cycle_rul
            # print(max(cycle_list), max_cycle_rul)

        # return kink_RUL(cycle_list,max_cycle)
        return self.kink_RUL(cycle_list, max_cycle)

    def kink_RUL(self, cycle_list, max_cycle):
        '''
        Piecewise linear function with zero gradient and unit gradient

                ^
                |
        MAXLIFE |-----------
                |            \
                |             \
                |              \
                |               \
                |                \
                |----------------------->
        '''
        knee_point = max_cycle - self.maxlife
        kink_RUL = []
        stable_life = self.maxlife
        # for i in range(0, len(cycle_list)):
        #     if i < knee_point:
        #         kink_RUL.append(MAXLIFE)
        #     else:
        #         tmp = kink_RUL[i - 1] - (stable_life / (max_cycle - knee_point))
        #         kink_RUL.append(tmp)
        # return kink_RUL
        if knee_point > 0:
            for i in range(0, len(cycle_list)):
                if i < knee_point:
                    kink_RUL.append(self.maxlife)
                else:
                    tmp = kink_RUL[i - 1] - (stable_life / (max_cycle - knee_point))
                    kink_RUL.append(tmp)
        else:
            for i in range(0, len(cycle_list)):
                kink_RUL.append(max_cycle - i - 1)
        return kink_RUL

if __name__=='__main__':
    # generate processed data
    # CMAPSS = get_CMAPSSData(
    #     file_path='../data/CMAPSSData/raw_data/',
    #     outputdir='../data/CMAPSSData/processed_data/',
    #     save_training_data=True,
    #     save_testing_data=True,
    #     files=[1, 2, 3, 4],
    #     MAXLIFE=120,
    #     min_max_norm='asymmetric'
    # )

    # read processed data
    read_CMAPSS = get_CMAPSSData(
        file_path='../data/CMAPSSData/raw_data/',
        outputdir='../data/CMAPSSData/processed_data/',
        save_training_data=False,
        save_testing_data=False,
        files=[1],
        MAXLIFE=120,
        min_max_norm='asymmetric'
    )
    training_data, testing_data, training_pd, testing_pd = read_CMAPSS.read_processed_data()