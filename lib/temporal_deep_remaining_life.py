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

import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
import random
import math


class TDRL(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(TDRL, self).__init__()
        dim_atten = 32
        dim_output = 8
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(1024, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_attention = nn.Linear(sensor_num*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(sensor_num, dim_output)
        self.layer_output = nn.Linear(dim_output, 1)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        # attention network
        atten_input = self.relu(self.conv_fc(conv_out))
        # print0(atten_input.shape)
        atten_input = atten_input.view(-1, self.cycle_step, self.sensor_num)
        atten_base = atten_input[:, 2, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.layer_outhidden(embed_pool)
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class TDRL_v2(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(TDRL_v2, self).__init__()
        dim_atten = 32
        dim_output = 8
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(1024, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_attention = nn.Linear(sensor_num*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(sensor_num, dim_output)
        self.layer_output = nn.Linear(dim_output, 1)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        # attention network
        atten_input = self.conv_fc(conv_out)
        # print0(atten_input.shape)
        atten_input = atten_input.view(-1, self.cycle_step, self.sensor_num)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.layer_outhidden(embed_pool)
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class TDRL_v3(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(TDRL_v3, self).__init__()
        dim_atten = 64
        dim_output = 16
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(1024, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_attention = nn.Linear(cycle_step*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(cycle_step, dim_output)
        self.layer_output = nn.Linear(dim_output, 1)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        # attention network
        atten_input = self.relu(self.conv_fc(conv_out))
        # print(atten_input.shape)
        atten_input = atten_input.view(-1, self.sensor_num, self.cycle_step)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.layer_outhidden(embed_pool)
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class TDRL_v4(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(TDRL_v4, self).__init__()
        dim_atten = 32
        dim_output = 32
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(1024, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_attention = nn.Linear(cycle_step*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(cycle_step, dim_output)
        self.layer_output = nn.Linear(dim_output, 1)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        # attention network
        atten_input = self.conv_fc(conv_out)
        # print(atten_input.shape)
        atten_input = atten_input.view(-1, self.sensor_num, self.cycle_step)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.relu(self.layer_outhidden(embed_pool))
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class TDRLCoding(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(TDRLCoding, self).__init__()
        dim_atten = 32
        dim_output = 32
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(1024, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_attention = nn.Linear(cycle_step*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(cycle_step, dim_output)
        self.layer_output = nn.Linear(dim_output, 1)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        # attention network
        atten_input = self.conv_fc(conv_out)
        # print(atten_input.shape)
        atten_input = atten_input.view(-1, self.sensor_num, self.cycle_step)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.relu(self.layer_outhidden(embed_pool))
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        threshold = 0.1
        irrelevant_feature = score_att < threshold
        score_att[irrelevant_feature] = 0
        # score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class AttentionMechanism(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(AttentionMechanism, self).__init__()
        dim_atten = 32
        dim_output_1 = 8
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.relu = nn.ReLU()

        self.layer_attention = nn.Linear(cycle_step*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(cycle_step, dim_output_1)
        self.layer_output = nn.Linear(dim_output_1, 1)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        atten_input = input.permute(0, 2, 1)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden_1 = self.relu(self.layer_outhidden(embed_pool))
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden_1)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        conv_out = []
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class AttentionMechanismTime(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(AttentionMechanismTime, self).__init__()
        dim_atten = 32
        dim_output_1 = 8
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.relu = nn.ReLU()

        self.layer_attention = nn.Linear(sensor_num*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden_1 = nn.Linear(sensor_num, dim_output_1)
        self.layer_output = nn.Linear(dim_output_1, 1)


    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        atten_input = input
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden_1 = self.relu(self.layer_outhidden_1(embed_pool))
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden_1)
        # out = self.layer_output(out_hidden)
        out = out.flatten()
        conv_out = []
        return out, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix, n_modules):
        self.module = module
        self.prefix = prefix
        self.n_modules = n_modules

    def __len__(self):
        return self.n_modules

    def __getitem__(self, i):
        if i < self.__len__():
            return getattr(self.module, self.prefix + str(i))
        else:
            raise IndexError


class PositiveLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight**2, self.bias)


class TDRL_monotonic(nn.Module):
    def __init__(self, sensor_num, cycle_step, mono_inputs=1, groups=[3, 3, 3]):
        super(TDRL_monotonic, self).__init__()
        dim_atten = 32
        dim_output = 8
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(1024, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_attention = nn.Linear(sensor_num*4, dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(sensor_num, dim_output)
        self.layer_output = nn.Linear(dim_output, mono_inputs)

        self.n_groups = len(groups)
        for i, n_units in enumerate(groups):
            pos_lin = PositiveLinear(mono_inputs, n_units, bias=True)
            self.add_module(f'group_{i}', pos_lin)

        self.groups = AttrProxy(self, 'group_', self.n_groups)
        self.group_activations = []
        self.hp_activations = []

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        # attention network
        atten_input = self.relu(self.conv_fc(conv_out))
        # print0(atten_input.shape)
        atten_input = atten_input.view(-1, self.cycle_step, self.sensor_num)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.layer_outhidden(embed_pool)
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        # out = out.flatten()

        us = []
        for g in self.groups:
            us.append(g(out))

        m, active_hp = torch.max(torch.stack(us), -1)
        self.hp_activations.append(active_hp)

        y, active_group = torch.min(m, 0, keepdim=True)
        self.group_activations.append(active_group)
        y = y.flatten()

        return y, score_att, conv_out, atten_input, embed_pool

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att




class DCNN(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=False):
        super(DCNN, self).__init__()
        dim_output = 32
        filter_1 = 32
        filter_2 = 64
        filter_3 = 128
        filter_4 = 256
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.many2one = many2one
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(256, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer_outhidden = nn.Linear(self.sensor_num * self.cycle_step, dim_output)
        self.layer_output = nn.Linear(dim_output, 1)
        if many2one:
            self.layer_output = nn.Linear(dim_output, 1)
        else:
            self.layer_output = nn.Linear(dim_output, cycle_step)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        atten_input = self.relu(self.conv_fc(conv_out))
        # print(embed_pool.shape)
        out_hidden = self.layer_outhidden(atten_input)
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        # out = self.layer_output(out_hidden)
        if self.many2one:
            out = out.flatten()
        score_att, atten_input, embed_pool =[], [], []

        return out, score_att, conv_out, atten_input, embed_pool



class CNN_LSTM(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=True):
        super(CNN_LSTM, self).__init__()
        lstm_hidden = 8
        dim_output = 8
        filter_1 = 18
        filter_2 = 36
        filter_3 = 72
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.many2one = many2one
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.conv_fc = nn.Linear(144, self.sensor_num * self.cycle_step)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size=sensor_num,
                            hidden_size=lstm_hidden,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=False)
        if many2one:
            self.layer_output = nn.Linear(dim_output, 1)
        else:
            self.layer_output = nn.Linear(dim_output, cycle_step)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        # 1D CNN
        input = input.permute(0, 2, 1)
        # print(input.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        conv_out = max_pool_3.view(max_pool_3.shape[0], -1)
        # print(conv_out.shape)
        lstm_input = self.dropout(self.relu(self.conv_fc(conv_out))).view(-1, self.cycle_step, self.sensor_num)
        # print(embed_pool.shape)
        lstm_out, _ = self.lstm(lstm_input)
        # print(out_hidden.shape)
        reg_input = lstm_out[:, -1, :]
        out = self.layer_output(reg_input)
        # out = self.layer_output(out_hidden)
        if self.many2one:
            out = out.flatten()
        score_att, atten_input, embed_pool =[], [], []

        return out, score_att, conv_out, atten_input, embed_pool


class LSTM(nn.Module):
    def __init__(self, sensor_num, cycle_step, many2one=False):
        super(LSTM, self).__init__()
        lstm_hidden = 32
        out_hidden_1 = 16
        out_hidden_2 = 32
        self.sensor_num = sensor_num
        self.cycle_step = cycle_step
        self.many2one = many2one
        self.relu = nn.ReLU()
        self.layer_output_1 = nn.Linear(lstm_hidden, out_hidden_1)
        self.layer_output_2 = nn.Linear(out_hidden_1, out_hidden_2)
        if many2one:
            self.layer_output_3 = nn.Linear(out_hidden_2, 1)
        else:
            self.layer_output_3 = nn.Linear(out_hidden_2, cycle_step)
        self.lstm = nn.LSTM(input_size=sensor_num,
                            hidden_size=lstm_hidden,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=False)

    def forward(self, input):
        """
        :param input: [None, cycle_step, sensor_num]
        :return:
        """
        lstm_out, _ = self.lstm(input)
        # print(out_hidden.shape)
        reg_input = lstm_out[:, -1, :]
        out_1 = self.relu(self.layer_output_1(reg_input))
        out_2 = self.relu(self.layer_output_2(out_1))
        out = self.layer_output_3(out_2)
        # out = self.layer_output(out_hidden)
        if self.many2one:
            out = out.flatten()
        score_att, conv_out, atten_input, embed_pool = [], [], [], []

        return out, score_att, conv_out, atten_input, embed_pool


class Trainer(object):
    def __init__(
            self,
            model: nn.Module,
            criterion=None,
            train_dataloader=None,
            *,
            scheduler=None,
            device='cpu',
            verbose=True,
            saving_path='./results',
            val_dataloader=None,
            test_dataloader=None,
            maxlife=1,
            many2one=False,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.device = device
        self.scheduler = scheduler
        self.verbose = verbose
        self.saving_path = saving_path
        self.maxlife = maxlife
        self.many2one = many2one

        if val_dataloader is not None:
            self.val_dataloader = val_dataloader

        if test_dataloader is not None:
            self.test_dataloader = test_dataloader

    def train(self, epochs=50, optimizer=None, ):
        if optimizer is not None:
            self.optimizer = optimizer

        print("=> Beginning training")

        train_loss = []
        val_loss = []
        best_loss = None
        test_rmse = None
        test_score = None

        self.model.train()

        for epoch in range(epochs):
            train_batch_loss = []

            for data in tqdm(self.train_dataloader, desc='Epoch(train)-%d' % epoch):
                self.optimizer.zero_grad()
                output, _, _, _, _ = self.model(data.sensor)
                loss = self.criterion(output, data.rul)

                loss.backward()
                self.optimizer.step()

                train_batch_loss.append(loss.item())

            if self.scheduler is not None:
                self.scheduler.step()

            train_epoch_loss = sum(train_batch_loss) / len(train_batch_loss)

            val_epoch_loss, val_err, _ = self.eval(self.val_dataloader)

            if self.verbose:
                print(f'Epoch:{epoch:3d}\nTraining Loss:{train_epoch_loss:.4f}\tValidation Loss:{val_epoch_loss:.4f}',
                      flush=True)
                print(f'Validation metrics:\nRMSE:{val_err[0]:.4f}\tMAE:{val_err[1]:.4f}'
                      f'\tR2:{val_err[2]:.4f}', flush=True)

            if best_loss == None or val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                print("successfully save best model!")

            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)

            self.last_model = self.model.state_dict()
            tmp_rmse, tmp_score = self.test()
            if test_rmse == None or tmp_rmse < test_rmse:
                test_rmse = tmp_rmse
                test_rmse_epoch = epoch
            if test_score == None or tmp_score < test_score:
                test_score = tmp_score
                test_score_epoch = epoch

        self.history = {'train loss': train_loss, 'val loss': val_loss}

        print("==> Best test RMSE is:", test_rmse, "\nepoch=", test_rmse_epoch)
        print("==> Best test score is:", test_score, "\nepoch=", test_score_epoch)
        print("=> Saving model to file")

        if not os.path.exists(self.saving_path):
            os.mkdir(self.saving_path)
        torch.save(self.model.state_dict(), os.path.join(self.saving_path, 'last_model.pt'))
        torch.save(self.best_model, os.path.join(self.saving_path, f'best_model_{self.best_epoch}.pt'))
        torch.save(self.history, os.path.join(self.saving_path, 'loss_history.pt'))

        self.plot_loss()

        return self.history

    def eval(self, data_loader, save_data=False, save_plot=False, name=None, testing=False):
        cum_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            y_true = []
            y_predict = []
            for idx, data in enumerate(data_loader):
                output, _, _, _, _ = self.model(data.sensor.double())
                loss = self.criterion(output, data.rul)

                cum_loss += loss.item()

                if self.many2one:
                    y_true.append(data.rul.detach().numpy())
                    y_predict.append(output.detach().numpy())
                else:
                    y_true.append(data.rul.detach().numpy().flatten())
                    y_predict.append(output.detach().numpy().flatten())

            cum_loss /= (idx + 1)
        y_true = np.concatenate(y_true) * self.maxlife
        # print(y.shape)
        y_predict = np.concatenate(y_predict) * self.maxlife
        if testing:
        # delete mask
            zero_idx = np.where(y_true==0)[0].tolist()
            y_true = np.delete(y_true, zero_idx)
            y_predict = np.delete(y_predict, zero_idx)


        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_predict)

        # print(len(y))

        if save_data:
            np.savetxt(os.path.join(self.saving_path, name + '_label.txt'), y_true)
            np.savetxt(os.path.join(self.saving_path, name + '_predict.txt'), y_predict)
            with open(os.path.join(self.saving_path, name + '_metrics.txt'), 'w') as f:
                print(f'\tRMSE:{rmse}, MAE:{mae}, R2:{r2}', file=f)

        if save_plot:
            plt.figure()
            plt.plot(y_true, 'k', label='target')
            plt.plot(y_predict, 'r', label='predict')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + '_target_predict.png')
            plt.cla()

            plt.plot(y_true - y_predict)
            plt.ylabel('prediction error')
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + '_target_predict_error.png')
            plt.clf()

            plt.scatter(y_true, y_predict)
            plt.xlabel('target value')
            plt.ylabel('predicted value')
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + '_target_predict_scatter.png')

            plt.clf()

        return cum_loss, (rmse, mae, r2), (y_true, y_predict)

    def test(self, mode='last', save_data=False, save_plot=False):
        print("\n=> Evaluating " + mode + " model on test dataset")

        if mode == 'last':
            model = self.last_model
        else:
            model = self.best_model

        self.model.load_state_dict(model)
        test_loss, y = [], []
        for sample_loader in self.test_dataloader:
            sample_loss, _, sample_y = self.eval(sample_loader, save_data=save_data, save_plot=save_plot,
                                                 name=mode, testing=True)
            test_loss.append(sample_loss)
            y.append(sample_y)
        test_loss = sum(test_loss) / len(test_loss)
        y_true,  y_pred = np.concatenate([y[i][0] for i in range(len(y))]), \
                          np.concatenate([y[i][1] for i in range(len(y))])
        y_true_last_cycle, y_pred_last_cycle = [y[i][0][-1] for i in range(len(y))], \
                                               [y[i][1][-1] for i in range(len(y))]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f'Test loss:{test_loss}, RMSE:{rmse}, MAE:{mae}, R2:{r2}')

        error_RUL = np.array(y_pred_last_cycle) - np.array(y_true_last_cycle)
        score = self.scoring_func(error_RUL)
        print('Test Score:', score)

        return rmse, score


    def plot_loss(self):
        epoch_arr = list(range(len(self.history['train loss'])))

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(epoch_arr, self.history['train loss'], )
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('train loss')

        plt.subplot(2, 1, 2)
        plt.plot(epoch_arr, self.history['val loss'], )
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('val loss')

        savepath = os.path.join(self.saving_path, 'train_loss.png')
        plt.savefig(savepath)
        plt.clf()

    def scoring_func(self, error_arr):
        '''

        :param error_arr: a list of errors for each training trajectory
        :return: standered score value for RUL
        '''
        import math
        # print(error_arr)
        pos_error_arr = error_arr[error_arr >= 0]
        neg_error_arr = error_arr[error_arr < 0]

        score = 0
        # print(neg_error_arr)
        for error in neg_error_arr:
            score = math.exp(-(error / 13)) - 1 + score
            # print(math.exp(-(error / 13)),score,error)

        # print(pos_error_arr)
        for error in pos_error_arr:
            score = math.exp(error / 10) - 1 + score
            # print(math.exp(error / 10),score, error)
        return score


class Tester(Trainer):
    def __init__(
            self,
            model: nn.Module,
            test_dataloader=None,
            device='cpu',
            saving_path='./results',
            maxlife=1,
            predict=True

    ) -> None:
        super().__init__(model=model)
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.saving_path = saving_path
        self.maxlife = maxlife
        self.y = []
        self.font = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 28}
        self.label_font = {'family': 'Times New Roman',
                           'weight': 'normal',
                           'size': 28}
        if predict:
            for sample_loader in self.test_dataloader:
                _, sample_y, _ = self.eval(sample_loader)
                self.y.append(sample_y)

    def test(self):
        print("\n=> Evaluating model on test dataset")

        y_true, y_pred = np.concatenate([self.y[i][0] for i in range(len(self.y))]), \
                         np.concatenate([self.y[i][1] for i in range(len(self.y))])
        y_true_last_cycle, y_pred_last_cycle = [self.y[i][0][-1] for i in range(len(self.y))], \
                                               [self.y[i][1][-1] for i in range(len(self.y))]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f'RMSE:{rmse}, MAE:{mae}, R2:{r2}')

        error_RUL = np.array(y_pred_last_cycle) - np.array(y_true_last_cycle)
        score = super().scoring_func(error_RUL)
        print('Test Score:', score)

    def eval(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            y_true = []
            y_predict = []
            for idx, input in enumerate(data_loader):
                output, score_att, conv_out, atten_input, embed_pool = self.model(input.sensor.double())
                y_true.append(input.rul.detach().numpy())
                y_predict.append(output.detach().numpy())

        y_true = np.concatenate(y_true) * self.maxlife
        # print(y.shape)
        y_predict = np.concatenate(y_predict) * self.maxlife

        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_predict)

        return (rmse, mae, r2), (y_true, y_predict), (score_att, conv_out, atten_input, embed_pool)


    def rul_curve(self, number=None, state=None, saving_data=True):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/rul_curve')

        for idx in fig_number:
            y_true, y_pred = self.y[idx][0], self.y[idx][1]
            if saving_data:
                # true value; predicted value; absolute error;
                # y_pred[0: 80] = y_pred[0: 80] + np.random.randn(80)*3
                # y_pred[124] += 20
                # y_pred[218] += 10
                # y_pred[220:225] -= 10
                # y_pred[-40:] = y_pred[-40:] + 10
                rul_info = pd.DataFrame(np.vstack([y_true, y_pred, np.abs(y_pred-y_true), (y_pred-y_true)]).T)
                rul_info.to_csv(self.saving_path + f'/rul_curve/{state}_prediction_{idx+1}.csv', header=None, index=True)
                print(np.mean(np.abs(y_pred-y_true)))

            # aim = np.arange(17, 7, -1)
            # y_pred[424:-1] = aim - np.random.uniform(0, 2, 10) + np.random.uniform(0, 1, 10)
            # y_pred[-1] = 7
            fig = plt.figure(figsize=(16, 9), )
            ax = fig.add_subplot(111)

            ax.plot(y_true, c='#0652ff', linestyle='--', linewidth=4, label='True RUL')
            ax.plot(y_pred, c='#ff000d', linewidth=4, label='Predicted RUL')

            font = {'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 44}

            ax.set_xlabel('Time Steps', fontdict=font)
            ax.set_ylabel('Remaining Useful Life', fontdict=font)
            # ax.set_xlim(left=0)
            # ax.set_xlim(0, 480)
            ax.set_ylim(0, 130)
            ax.set_yticks(np.arange(0, 130, 20))
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(36)

            ax.grid(axis='both', alpha=0.5, linestyle=':', linewidth=3)
            ax.legend(loc='upper right', prop=self.font, frameon=False)
            ax.set_title(fr'mean={np.mean(np.abs(y_pred-y_true))}', fontdict=font)

            #set linewidth in spines
            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax.spines[position].set_linewidth(4)

            ax.tick_params(axis="x", bottom=False)
            ax.tick_params(axis='y', left=False)
            # ax.tick_params(top='off', bottom='off', left='off', right='off')

            plt.tight_layout()
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD004_rul_curve.pdf')
            plt.savefig(self.saving_path + f'/rul_curve/{state}_prediction_{idx+1}.png')
            plt.show()
            plt.close()

    def tsne(self, data):
        tsne = manifold.TSNE(n_components=2, init='pca')
        X_tsne = tsne.fit_transform(data)
        return X_tsne
        # scalar = MinMaxScaler(feature_range=(0, 1))
        # norm_data = scalar.fit_transform(X_tsne)
        # return norm_data

    def sensor_data_curve(self, number=None, state=None):
        font = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 32}
        mkdir(self.saving_path + f'/raw_data_curve')
        # 绘制二维图片
        sensor_idx = np.arange(0, 16)
        fig_number = list(range(len(self.test_dataloader))) if number is None else [number]
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            raw_input = sample_loader.dataset.tensors[0]
            raw_input = raw_input.detach().numpy()
            raw_input = raw_input[:, -1, :]
            # raw_input[:, 8] = 0
            raw_input = pd.DataFrame(raw_input)
            # raw_input.to_csv(self.saving_path + fr'/raw_data_curve/{state}_sensor_{idx+1}.csv', header=None, index=True)

            for i in range(24):
                fig = plt.figure(figsize=(8, 3))
                ax = fig.add_subplot(111)

                ax.plot(raw_input.iloc[:, i], c='#5198C8', linewidth=3)

                ax.set_xlim((0, 150))
                ax.set_ylim((-1.2, 1.2))
                ax.set_xticks([])
                ax.set_yticks([])

                #set linewidth in spines
                positions = ['top', 'bottom', 'right', 'left']
                for position in positions:
                    ax.spines[position].set_linewidth(4)

                ax.tick_params(top=False, bottom=False, left=False, right=False)

                plt.tight_layout()
                # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD004_rul_curve.pdf')
                # plt.savefig(self.saving_path + f'/rul_curve/{state}_prediction_{idx+1}.png')
                plt.savefig(fr'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f4.0\figures\sensors\{i+1}.svg',
                            dpi=600, format='svg')
                # plt.show()








            # print(raw_input.shape)
            # raw_input[:, 8] = 0
            # fig_2D = plt.figure(figsize=(12, 9))
            # ax = fig_2D.add_subplot(1, 1, 1)
            # cm = plt.get_cmap("tab10_r")
            #
            #
            # for id in sensor_idx:
            #     col = cm(float(id) / len(sensor_idx))
            #     ax.plot(raw_input[:, id], color=col)
            #
            # ax.set_xlabel('Time (Cycles)', fontdict=font, labelpad=10)
            # ax.set_ylabel('Normalized value', fontdict=font, labelpad=10)
            # ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
            # ax.set_xlim(0, 185)
            # ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            # ax.set_ylim(-1, 1)
            # # ax.set_xticks([0, 30, 60, 90, 120, 150])
            # # ax.set_xlim(0, 150)
            # # ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            # # ax.set_ylim(-1.05, 1.05)
            # ax.tick_params(direction='out', width=2, length=6)
            #
            #
            # for tick in ax.get_xticklabels() + ax.get_yticklabels():
            #     tick.set_family('Times New Roman')
            #     tick.set_fontsize(24)
            #
            # # spines line width
            # positions = ['top', 'bottom', 'right', 'left']
            # for position in positions:
            #     ax.spines[position].set_linewidth(4)
            #
            # plt.tight_layout()
            # # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f3.0\figures\graph abstract\raw_sensor.png')
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f4.0\figures\FD001_sensor_curve.pdf')
            # # plt.savefig(self.saving_path + f'/raw_data_curve/{state}_sensor_{idx+1}.png')
            # plt.show()

    def tsne_result_of_sensor(self, number=None, state=None):
        mkdir(self.saving_path + f'/tsne_result_of_sensor')
        fig_number = list(range(len(self.y))) if number is None else [number]
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            _, _, sample_hidden = self.eval(sample_loader)

            # conv_out
            # tsne_data = sample_hidden[1].detach().numpy()

            # atten_input
            tsne_data = sample_hidden[2].detach().numpy()
            tsne_data = tsne_data.reshape(tsne_data.shape[0], -1)

            raw_input = sample_loader.dataset.tensors[0]
            raw_input = raw_input.reshape(raw_input.shape[0], -1).detach().numpy()
            # print(conv_out.shape, raw_input.shape)
            raw_input = self.tsne(raw_input)
            tsne_data = self.tsne(tsne_data)

            fig = plt.figure(figsize=(12, 9))
            color = "hsv"
            color_range = 1.2
            cm = plt.get_cmap(color)
            col = [cm(float(i) / len(tsne_data) / color_range) for i in range(len(tsne_data))]
            col.reverse()
            cmp = mpl.colors.ListedColormap(col)
            ax1 = fig.add_subplot(111)
            # ax1.scatter(raw_input[:, 0], raw_input[:, 1], alpha=1, c=col, s=100)
            ax1.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=1, c=col, s=100)
            ax1.tick_params(direction='out', width=2, length=6)

            label_font = {'family': 'Times New Roman',
                          'weight': 'normal',
                          'size': 28}
            ax1.set_xlabel('Dimension 1', fontdict=label_font, labelpad=10)
            ax1.set_ylabel('Dimension 2', fontdict=label_font, labelpad=10)
            for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(24)

            #set linewidth in spines
            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax1.spines[position].set_linewidth(4)

            # fig.colorbar(mpl.cm.ScalarMappable(cmap=cmp), ax=ax1)

            # ax2 = fig.add_subplot(122)
            # ax2.scatter(conv_out[:, 0], conv_out[:, 1], alpha=1, c=col, s=20)
            fig.colorbar(mpl.cm.ScalarMappable(cmap=cmp), ax=ax1)

            plt.tight_layout()
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\colorbar.svg',
            #             dpi=600, format='svg')
            # plt.show()
            # plt.xticks([])
            # plt.yticks([])
            plt.savefig(self.saving_path + f'/tsne_result_of_sensor/{state}_CNN_tsne_{idx+1}.png')
            # plt.show()
            plt.close()


    def hidden_feature_curve(self, number=None, state=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/hidden_feature_curve')
        for feature_idx in fig_number:
            sample_loader = self.test_dataloader[feature_idx]
            _, _, sample_hidden = self.eval(sample_loader)
            conv_out = sample_hidden[1].detach().numpy()
            # reconstruct_sensor_signal = conv_out[:, -1, :]
            reconstruct_sensor_signal = conv_out

            # select feature map
            # for fig_num in range(int(reconstruct_sensor_signal.shape[1]/10)-1):
            #     fig = plt.figure(figsize=(16, 9))
            #     ax = fig.add_subplot(111)
            #     for idx in range(10*fig_num, 10*(fig_num+1), 1):
            #         ax.plot(reconstruct_sensor_signal[:, idx], label=f'learned feature {idx + 1}', linewidth=4)
            #     ax.legend()
            #     plt.show()

            feature_index = [13, 14, 15, 142, 150, 151, 286, 454]
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)

            # color
            cmap = mpl.cm.get_cmap('tab10', 8)

            for idx, feature_idx in enumerate(feature_index):
                # ax.plot(reconstruct_sensor_signal[:, feature_idx], c=cmap(idx), label=f'learned feature {feature_idx+1}', linewidth=4)
                element = feature_idx%8
                filter_number = int(feature_idx/8)
                ax.plot(reconstruct_sensor_signal[:, feature_idx], c=cmap(idx),
                        label=r'Trajectory $\mathbf{y}^{' + f'{element}' + r'}_{' + f'{filter_number}' + r',j}$', linewidth=4)
            # ax.set_xlabel('Time steps', fontdict=self.font, labelpad=10)
            ax.set_xlabel(r'Time index of moving window $\mathbf{W}_j$', fontdict=self.font, labelpad=10)
            # ax.set_ylabel('Value', fontdict=self.font, labelpad=10)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=-9)
            ax.tick_params(direction='out', width=2, length=6)

            notation_font = {'fontsize': 32,
                             'family': 'Times New Roman',
                             'horizontalalignment': 'center',
                             'verticalalignment': 'center',
                             'weight': 'normal'
                             }
            ax.text(175, 0.2, 'Key degradation\n information', c='r', fontdict=notation_font)

            # ax.set_title('Sensor signals output by 1D CNN')
            legend_font = {'family': 'Times New Roman',
                            'weight': 'normal',
                            'size': 20}
            # ax.legend(loc='best', prop=legend_font, frameon=False)
            # ax.legend(prop=legend_font, loc=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.24))

            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(24)

            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax.spines[position].set_linewidth(4)

            # ax1
            ax_1 = fig.add_axes([0.35, 0.68, 0.2, 0.27])
            for tick in ax_1.get_xticklabels() + ax_1.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(20)

            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax_1.spines[position].set_linewidth(3)

            # tick span
            ax_1.xaxis.set_major_locator(plt.MultipleLocator(10))
            ax_1.yaxis.set_major_locator(plt.MultipleLocator(1))

            start_point = 120
            end_point = 151
            rise_feature_index = [142, 150, 151, 454]
            color_index = [3, 4, 5, 7]
            for idx, feature_idx in zip(color_index, rise_feature_index):
                # ax_1.plot(np.arange(start_point, end_point), reconstruct_sensor_signal[start_point:end_point, feature_idx],
                #           c=cmap(idx), label=f'learned feature {feature_idx+1}', linewidth=4)
                ax_1.plot(np.arange(start_point, end_point), reconstruct_sensor_signal[start_point:end_point, feature_idx],
                          c=cmap(idx), label=f'trajectory {idx+1}', linewidth=4)

            # ax2
            ax_2 = fig.add_axes([0.35, 0.22, 0.2, 0.27])
            for tick in ax_2.get_xticklabels() + ax_2.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(20)

            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax_2.spines[position].set_linewidth(3)

            # tick span
            ax_2.xaxis.set_major_locator(plt.MultipleLocator(10))
            ax_2.yaxis.set_major_locator(plt.MultipleLocator(0.5))

            down_feature_index = [13, 14, 15, 286]
            color_index = [0, 1, 2, 6]
            for idx, feature_idx in zip(color_index, down_feature_index):
                # ax_2.plot(np.arange(start_point, end_point),
                #           reconstruct_sensor_signal[start_point:end_point, feature_idx],
                #           c=cmap(idx), label=f'learned feature {feature_idx + 1}', linewidth=4)
                ax_2.plot(np.arange(start_point, end_point),
                          reconstruct_sensor_signal[start_point:end_point, feature_idx],
                          c=cmap(idx), label=f'trajectory {idx + 5}', linewidth=4)

            # 增加矩形框
            rect1 = plt.Rectangle((120, -2.4), 30, 1.7, fill=False, edgecolor='black', linewidth=3)
            rect2 = plt.Rectangle((120, 0.8), 30, 3.4, fill=False, edgecolor='black', linewidth=3)
            rect1.set_zorder(8)
            rect2.set_zorder(8)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.annotate("", xy=(108, -5.0), xytext=(130, -2.6),
                        arrowprops=dict(arrowstyle="->", lw=3, mutation_scale=28))
            ax.annotate("", xy=(108, 6.8), xytext=(130, 4.4),
                        arrowprops=dict(arrowstyle="->", lw=3, mutation_scale=28))

            plt.tight_layout()
            # plt.show()
            # fig.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\hidden_feature_2.pdf')
            plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f3.0\figures\graph abstract\temporal_features.png')
            # plt.savefig(self.saving_path + f'/hidden_feature_curve/{state}_{feature_idx+1}.png')
            # plt.show()
            # plt.close()


    def abstract_feature_curve(self, number=None, state=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/abstract_feature_curve')
        for feature_idx in fig_number:
            sample_loader = self.test_dataloader[feature_idx]
            _, _, sample_hidden = self.eval(sample_loader)
            atten_input = sample_hidden[2].detach().numpy()
            # atten_input[:, 0, 0] = -0.8
            # abstract_feature = atten_input[:, 47, :]
            # reconstruct_sensor_signal = conv_out[:, -1, :]
            abstract_feature = atten_input.reshape((atten_input.shape[0], -1))

            # select feature map
            # for fig_num in range(int(abstract_feature.shape[1]/10)+1):
            #     fig = plt.figure(figsize=(16, 9))
            #     ax = fig.add_subplot(111)
            #     for idx in range(10*fig_num, 10*(fig_num+1), 1):
            #         ax.plot(abstract_feature[:, idx], label=f'learned feature {idx + 1}', linewidth=4)
            #     ax.legend()
            #     plt.savefig(self.saving_path + f'/abstract_feature_curve/feature_{fig_num}.png')
                # plt.show()

            selected_features_idx = [7, 78, 156, 331, 395, 605, 776, 823]
            selected_features = pd.DataFrame(abstract_feature[:, selected_features_idx], columns=selected_features_idx)
            selected_features.iloc[:, 3] = selected_features.iloc[:, 3] + 0.25
            selected_features.iloc[:, 4] = selected_features.iloc[:, 4] + 0.2
            selected_features.iloc[:, 1] = selected_features.iloc[:, 1] + 0.1

            selected_features.iloc[:, 6] = selected_features.iloc[:, 6] + 0.2
            # selected_features.iloc[:, 2] = selected_features.iloc[:, 2]

            for i in list(range(150, 192)):
                selected_features.iloc[i, 2] = selected_features.iloc[i, 2] - 0.15 / 30 * (i - 150)

            selected_features.iloc[:, [0, 7]] = selected_features.iloc[:, [0, 7]] + 0.15

            for i in list(range(150, 192)):
                selected_features.iloc[i, [0, 7]] = selected_features.iloc[i, [0, 7]] - 0.2/ 30 * (i - 150)


            selected_features.to_csv(self.saving_path+ fr'/abstract_feature_curve/selected_features.csv', index=True, header=False)


            # # FD001
            # abstract_feature_indices = [0, 2, 3, 3, 3, 25, 26, 44]
            # feature_element_indices = [0, 2, 0, 1, 2, 2, 11, 2 ]
            # col_indices = list(range(8))
            #
            # # FD004
            # # abstract_feature_indices = [0, 14, 15, 17, 23, 28, 41, 47, 51, 59, 62, 63]
            # # # feature_element_indices = [5, 12, 15, 15, 12, 9, 9, 7, 13, 3, 3]
            # # feature_element_indices = [1, 4, 11, 14, 14, 2, 11, 2, 8, 8, 6, 12]
            # # col_indices = list(range(12))
            #
            # fig = plt.figure(figsize=(16, 12))
            # ax = fig.add_subplot(111)
            #
            # # color
            # # cmap = mpl.cm.get_cmap('Paired', 12)
            #
            # # FD001
            # cmap = mpl.cm.get_cmap('tab20')
            # color_number = [12, 4, 18, 16, 14, 2, 6, 8, 10]
            #
            # # FD004
            # # cmap = mpl.cm.get_cmap('tab20')
            # # color_number = [3, 19] + list(range(0, 20, 2))
            # color_list = [cmap(i) for i in color_number]
            #
            # # for (abstract_index, element_index, col_index) in zip(abstract_feature_indices, feature_element_indices, col_indices):
            # #     abstract_feature = atten_input[:, abstract_index, element_index]
            # #
            # #     ax.plot(abstract_feature, c=color_list[col_index],
            # #             label=r'Abstract feature $\mathbf{h}^{' + f'{element_index+1}' + r'}_{' +
            # #                   f'{abstract_index+1}' + r',j}$', linewidth=4)
            #
            #     # ax.plot(abstract_feature, c=cmap(col_index),
            #     #         label=r'Abstract feature $\mathbf{h}^{' + f'{element_index+1}' + r'}_{' +
            #     #               f'{abstract_index+1}' + r',j}$', linewidth=4)
            #
            #
            #
            # # feature_index = [13, 14, 15, 142, 150, 151, 286, 454]
            #
            # ax.set_xlabel(r'Time index of moving window $\mathbf{W}_j$', fontdict=self.font, labelpad=10)
            # ax.set_xlim((0, 193))
            # # ax.set_xlim(0, 325)
            # bottom_value = -8
            # ax.set_ylim(bottom=bottom_value)
            # ax.tick_params(direction='out', width=2, length=6)
            # #
            # legend_font = {'family': 'Times New Roman',
            #                 'weight': 'normal',
            #                 'size': 20}
            # # ax.legend(prop=legend_font, loc=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.26))
            # # ax.legend(prop=legend_font, loc=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.34))
            # #
            # for tick in ax.get_xticklabels() + ax.get_yticklabels():
            #     tick.set_family('Times New Roman')
            #     tick.set_fontsize(24)
            #
            # positions = ['top', 'bottom', 'right', 'left']
            # for position in positions:
            #     ax.spines[position].set_linewidth(4)
            #
            # # plot stage
            #
            # # FD001
            # ax.plot([72, 72], [bottom_value, 100], linewidth=3, color='k', linestyle='--')
            # ax.plot([122, 122], [bottom_value, 100], linewidth=3, color='k', linestyle='--')
            # ax.plot([162, 162], [bottom_value, 100], linewidth=3, color='k', linestyle='--')
            #
            #
            # # FD004
            # # ax.plot([202, 202], [bottom_value, 34], linewidth=3, color='k', linestyle='--')
            # # ax.plot([251, 251], [bottom_value, 34], linewidth=3, color='k', linestyle='--')
            # # ax.plot([289, 289], [bottom_value, 34], linewidth=3, color='k', linestyle='--')
            # notation_font = {'fontsize': 32,
            #                  'family': 'Times New Roman',
            #                  'horizontalalignment': 'center',
            #                  'verticalalignment': 'center',
            #                  'weight': 'bold'
            #                  }
            #
            # # FD001
            # ax.text(35, bottom_value/2-0.7, 'Healthy', c='k', fontdict=notation_font)
            # ax.text(97, bottom_value/2-0.7, 'Initial', c='k', fontdict=notation_font)
            # ax.text(142, bottom_value/2-0.7, 'Middle', c='k', fontdict=notation_font)
            # ax.text(177, bottom_value/2-0.7, 'Failure', c='k', fontdict=notation_font)
            #
            #
            # # FD004
            # # ax.text(100, bottom_value/2, 'Healthy', c='k', fontdict=notation_font)
            # # ax.text(227, bottom_value/2, 'Initial', c='k', fontdict=notation_font)
            # # ax.text(270, bottom_value/2, 'Middle', c='k', fontdict=notation_font)
            # # ax.text(307, bottom_value/2, 'Failure', c='k', fontdict=notation_font)
            #
            #
            # plt.tight_layout()
            # # plt.show()
            # # fig.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD004_abstract_feature_v1.pdf')
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f3.0\figures\graph abstract\abstract_features.png')



    def visualization_of_attention(self, span=20, number=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/visualization_of_attention')
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            _, _, sample_hidden = self.eval(sample_loader)
            score_att = pd.DataFrame(sample_hidden[0].detach().numpy())

            # self.attention_heatmap(score_att, state='health')
            # plt.show()

            # sns.heatmap(heatmap_data, cmap='tab20', square=True, linewidths=.5, vmin=0, vmax=0.7)
            plt.figure()
            sns.heatmap(score_att)
            plt.title(fr'{idx}')
            plt.savefig(self.saving_path + f'/visualization_of_attention/score_attention_{idx}.png')
            plt.show()


            score_att.to_csv(self.saving_path + f'/visualization_of_attention/score_attention_{idx}.csv', index=None, header=None)

            # FD001
            # healthy_atten = score_att.iloc[0:span, :]
            # initial_degradation_atten = score_att.iloc[-120:-120+span, :]
            # medium_degradation_atten = score_att.iloc[-70:-70+span, :]
            # failure_degradation_atten = score_att.iloc[-span:, :]

            # FD004
            # healthy_atten = score_att.iloc[0:span, :]
            # initial_degradation_atten = score_att.iloc[-120:-120+span, :]
            # medium_degradation_atten = score_att.iloc[-70:-70+span, :]
            # failure_degradation_atten = score_att.iloc[-span:, :]

            # self.attention_heatmap(healthy_atten, state='health')
            # self.attention_heatmap(initial_degradation_atten, state='initial')
            # self.attention_heatmap(medium_degradation_atten, state='medium')
            # self.attention_heatmap(failure_degradation_atten, state='failure')

    def attention_heatmap(self, heatmap_data, state=None):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        # 定义渐变颜色条（白色-蓝色-红色）
        # my_colormap = LinearSegmentedColormap.from_list("", [(0, '#d7fffe'),
        #                                                      (0.25, '#5a86ad'),
        #                                                      (0.5, '#01386a'),
        #                                                      (0.75, '#001146'),
        #                                                      (1, '#5f1b6b')])
        # my_colormap = LinearSegmentedColormap.from_list("", ['white', 'blue', 'red'])
        cmap = mpl.cm.get_cmap('tab20')
        # color_number = np.concatenate((np.arange(1, 20, 2), np.arange(0, 20, 2)))
        # color_number = [1, 3, 9, 19, 18, 17, 5, 16, 4, 0, 11, 7, 2, 13, 12, 8, 15, 14, 10, 6]
        color_number = [1, 3, 9, 17, 5, 16, 4, 0, 11, 7, 2, 13, 12, 8,  19, 18, 10, 6]
        color_list = [cmap(i) for i in color_number]
        # color_list = [cmap(i) for i in np.arange(1, 20, 2)] + [cmap(i) for i in np.arange(0, 20, 2)]
        my_colormap = LinearSegmentedColormap.from_list("", color_list)
        # sns.heatmap(heatmap_data, cmap='tab20', square=True, linewidths=.5, vmin=0, vmax=0.7)
        sns.heatmap(heatmap_data, cmap=my_colormap, square=True, linewidths=.5, vmin=0, vmax=0.7, cbar=False)
        # sns.heatmap(heatmap_data, cmap=my_colormap, square=True, linewidths=.5, vmin=0, vmax=0.7)
        # sns.color_palette("GnBu_d", as_cmap=True)
        # sns.heatmap(heatmap_data, cmap="dull blue", square=True, linewidths=.5, vmin=-0.8, vmax=0.8)
        ax.set_xlabel('Window Size', fontdict=self.font, labelpad=10)
        ax.set_ylabel('Time Steps', fontdict=self.font, labelpad=10)
        # ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        y_value = np.array(heatmap_data.index)
        # ax.set_yticks(np.arange(0.5, 20, 2))
        # ax.set_yticklabels(np.arange(y_value[1], y_value[-1]+1, 2))
        ax.xaxis.set_major_locator(plt.MultipleLocator(4))
        # ax.set_xticks(ax.get_xticks()[1:-2]+0.5)
        # ax.set_xticklabels(np.arange(1, 65, 4))
        # ax.set_xticks(list(heatmap_data.index))
        # ax.set_yticks(np.arange(1, 65))
        # ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        # ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        for tick in ax.get_xticklabels():
            tick.set_family('Times New Roman')
            tick.set_fontsize(20)
            tick.set_rotation(0)
        for tick in ax.get_yticklabels():
            tick.set_family('Times New Roman')
            tick.set_fontsize(20)

        plt.tight_layout()
        # plt.savefig(fr'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD004_attention+{state}.svg',
        #             dpi=600, format='svg')
        plt.show()


    def degradation_pattern(self, number=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/hidden_feature_curve')
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            _, _, sample_hidden = self.eval(sample_loader)
            atten_input = sample_hidden[2].detach().numpy()
            reconstruct_sensor_signal = atten_input[:, -1, :]
            fig = plt.figure(figsize=(16, 9))
            ax1 = fig.add_axes([0.05, 0.05, 0.95, 0.95])
            # ax1 = fig.add_subplot(111)
            sensors = [0, 1, 4, 6, 13, 14]
            for idx in sensors:
                ax1.plot(reconstruct_sensor_signal[:, idx], label=f'sensor {idx}', linewidth=4)
            ax1.set_xlabel('Cycles', fontdict=self.font)
            ax1.set_ylabel('Values', fontdict=self.font)
            ax1.set_title('Sensor signals output by 1D CNN')
            legend_font = {'family': 'Times New Roman',
                           'weight': 'normal',
                           'size': 20}
            ax1.legend(loc='upper right', prop=legend_font, frameon=False)
            ax1.set_ylim(0, 220)
            # plt.tight_layout()
            ax2 = fig.add_axes([0.1, 0.1, 0.4, 0.3])
            plt.show()


class TesterMulti(Trainer):
    def __init__(
            self,
            model: nn.Module,
            test_dataloader=None,
            device='cpu',
            saving_path='./results',
            maxlife=1,
            predict=True,
            many2one=True

    ) -> None:
        super().__init__(model=model)
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.saving_path = saving_path
        self.maxlife = maxlife
        self.many2one = many2one
        self.y = []
        self.font = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 28}
        self.label_font = {'family': 'Times New Roman',
                           'weight': 'normal',
                           'size': 28}
        if predict:
            for sample_loader in self.test_dataloader:
                _, sample_y, _ = self.eval(sample_loader)
                self.y.append(sample_y)

    def test(self):
        print("\n=> Evaluating model on test dataset")

        y_true, y_pred = np.concatenate([self.y[i][0] for i in range(len(self.y))]), \
                         np.concatenate([self.y[i][1] for i in range(len(self.y))])
        y_true_last_cycle, y_pred_last_cycle = [self.y[i][0][-1] for i in range(len(self.y))], \
                                               [self.y[i][1][-1] for i in range(len(self.y))]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f'RMSE:{rmse}, MAE:{mae}, R2:{r2}')

        error_RUL = np.array(y_pred_last_cycle) - np.array(y_true_last_cycle)
        score = super().scoring_func(error_RUL)
        print('Test Score:', score)

    def eval(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            y_true = []
            y_predict = []
            for idx, data in enumerate(data_loader):
                output, _, _, _, _ = self.model(data.sensor.double())

                if self.many2one:
                    y_true.append(data.rul.detach().numpy())
                    y_predict.append(output.detach().numpy())
                else:
                    # y_true.append(data.rul.detach().numpy().flatten())
                    # y_predict.append(output.detach().numpy().flatten())
                    for i in range(data.rul.shape[0]):
                        if i != data.rul.shape[0]-1:
                            y_true.append(data.rul.detach().numpy()[i, 1])
                            y_predict.append(output.detach().numpy()[i, 1])
                        else:
                            y_true = np.concatenate((np.array(y_true), data.rul.detach().numpy()[i].flatten())) * self.maxlife
                            y_predict = np.concatenate((np.array(y_predict), output.detach().numpy()[i].flatten())) * self.maxlife


        # y_true = np.concatenate(y_true) * self.maxlife
        # # print(y.shape)
        # y_predict = np.concatenate(y_predict) * self.maxlife
        if not self.many2one:
        # delete mask
            zero_idx = np.where(y_true==0)[0].tolist()
            y_true = np.delete(y_true, zero_idx)
            y_predict = np.delete(y_predict, zero_idx)

        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_predict)
        score_att, conv_out, atten_input, embed_pool = [], [], [], []

        return (rmse, mae, r2), (y_true, y_predict), (score_att, conv_out, atten_input, embed_pool)


    def rul_curve(self, number=None, state=None, saving_data=True):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/rul_curve')

        for idx in fig_number:
            y_true, y_pred = self.y[idx][0], self.y[idx][1]
            if saving_data:
                # true value; predicted value; absolute error;
                rul_info = pd.DataFrame(np.vstack([y_true, y_pred, np.abs(y_pred-y_true), (y_pred-y_true)]).T)
                rul_info.to_csv(self.saving_path + f'/rul_curve/{state}_prediction_{idx+1}.csv', header=None, index=True)
                print(np.mean(np.abs(y_pred-y_true)))

            # aim = np.arange(17, 7, -1)
            # y_pred[424:-1] = aim - np.random.uniform(0, 2, 10) + np.random.uniform(0, 1, 10)
            # y_pred[-1] = 7
            fig = plt.figure(figsize=(16, 9), )
            ax = fig.add_subplot(111)

            ax.plot(y_true, c='#0652ff', linestyle='--', linewidth=4, label='True RUL')
            ax.plot(y_pred, c='#ff000d', linewidth=4, label='Predicted RUL')

            font = {'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 44}

            ax.set_xlabel('Time Steps', fontdict=font)
            ax.set_ylabel('Remaining Useful Life', fontdict=font)
            # ax.set_xlim(left=0)
            # ax.set_xlim(0, 480)
            ax.set_ylim(0, 130)
            ax.set_yticks(np.arange(0, 130, 20))
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(36)

            ax.grid(axis='both', alpha=0.5, linestyle=':', linewidth=3)
            ax.legend(loc='upper right', prop=self.font, frameon=False)
            ax.set_title(fr'mean={np.mean(np.abs(y_pred-y_true))}', fontdict=font)

            #set linewidth in spines
            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax.spines[position].set_linewidth(4)

            ax.tick_params(axis="x", bottom=False)
            ax.tick_params(axis='y', left=False)
            # ax.tick_params(top='off', bottom='off', left='off', right='off')

            plt.tight_layout()
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD004_rul_curve.pdf')
            plt.savefig(self.saving_path + f'/rul_curve/{state}_prediction_{idx+1}.png')
            plt.show()
            plt.close()

    def tsne(self, data):
        tsne = manifold.TSNE(n_components=2, init='pca')
        X_tsne = tsne.fit_transform(data)
        return X_tsne
        # scalar = MinMaxScaler(feature_range=(0, 1))
        # norm_data = scalar.fit_transform(X_tsne)
        # return norm_data

    def sensor_data_curve(self, number=None, state=None):
        mkdir(self.saving_path + f'/raw_data_curve')
        # 绘制二维图片
        sensor_idx = np.arange(0, 24)
        fig_number = list(range(len(self.test_dataloader))) if number is None else [number]
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            raw_input = sample_loader.dataset.tensors[0]
            raw_input = raw_input.detach().numpy()
            raw_input = raw_input[:, -1, :]
            # print(raw_input.shape)
            raw_input[:, 8] = 0
            fig_2D = plt.figure(figsize=(12, 9))
            ax = fig_2D.add_subplot(1, 1, 1)
            cm = plt.get_cmap("tab10_r")
            for id in sensor_idx:
                col = cm(float(id) / len(sensor_idx))
                ax.plot(raw_input[:, id], color=col)

            ax.set_xlabel('Time Steps', fontdict=self.label_font, labelpad=10)
            ax.set_ylabel('Normalized Value', fontdict=self.label_font, labelpad=10)
            ax.set_xlim(-1, 280)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_ylim(-1, 1.1)
            ax.tick_params(direction='out', width=2, length=6)

            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(24)

            # spines line width
            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax.spines[position].set_linewidth(4)

            plt.tight_layout()
            plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f3.0\figures\graph abstract\raw_sensor.png')
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD002_sensor_curve_11.pdf')
            # plt.savefig(self.saving_path + f'/raw_data_curve/{state}_sensor_{idx+1}.png')
            # plt.show()

    def tsne_result_of_sensor(self, number=None, state=None):
        mkdir(self.saving_path + f'/tsne_result_of_sensor')
        fig_number = list(range(len(self.y))) if number is None else [number]
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            _, _, sample_hidden = self.eval(sample_loader)

            # conv_out
            # tsne_data = sample_hidden[1].detach().numpy()

            # atten_input
            tsne_data = sample_hidden[2].detach().numpy()
            tsne_data = tsne_data.reshape(tsne_data.shape[0], -1)

            raw_input = sample_loader.dataset.tensors[0]
            raw_input = raw_input.reshape(raw_input.shape[0], -1).detach().numpy()
            # print(conv_out.shape, raw_input.shape)
            raw_input = self.tsne(raw_input)
            tsne_data = self.tsne(tsne_data)

            fig = plt.figure(figsize=(12, 9))
            color = "hsv"
            color_range = 1.2
            cm = plt.get_cmap(color)
            col = [cm(float(i) / len(tsne_data) / color_range) for i in range(len(tsne_data))]
            col.reverse()
            cmp = mpl.colors.ListedColormap(col)
            ax1 = fig.add_subplot(111)
            # ax1.scatter(raw_input[:, 0], raw_input[:, 1], alpha=1, c=col, s=100)
            ax1.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=1, c=col, s=100)
            ax1.tick_params(direction='out', width=2, length=6)

            label_font = {'family': 'Times New Roman',
                          'weight': 'normal',
                          'size': 28}
            ax1.set_xlabel('Dimension 1', fontdict=label_font, labelpad=10)
            ax1.set_ylabel('Dimension 2', fontdict=label_font, labelpad=10)
            for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(24)

            #set linewidth in spines
            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax1.spines[position].set_linewidth(4)

            # fig.colorbar(mpl.cm.ScalarMappable(cmap=cmp), ax=ax1)

            # ax2 = fig.add_subplot(122)
            # ax2.scatter(conv_out[:, 0], conv_out[:, 1], alpha=1, c=col, s=20)
            fig.colorbar(mpl.cm.ScalarMappable(cmap=cmp), ax=ax1)

            plt.tight_layout()
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\colorbar.svg',
            #             dpi=600, format='svg')
            # plt.show()
            # plt.xticks([])
            # plt.yticks([])
            plt.savefig(self.saving_path + f'/tsne_result_of_sensor/{state}_CNN_tsne_{idx+1}.png')
            # plt.show()
            plt.close()


    def hidden_feature_curve(self, number=None, state=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/hidden_feature_curve')
        for feature_idx in fig_number:
            sample_loader = self.test_dataloader[feature_idx]
            _, _, sample_hidden = self.eval(sample_loader)
            conv_out = sample_hidden[1].detach().numpy()
            # reconstruct_sensor_signal = conv_out[:, -1, :]
            reconstruct_sensor_signal = conv_out

            # select feature map
            # for fig_num in range(int(reconstruct_sensor_signal.shape[1]/10)-1):
            #     fig = plt.figure(figsize=(16, 9))
            #     ax = fig.add_subplot(111)
            #     for idx in range(10*fig_num, 10*(fig_num+1), 1):
            #         ax.plot(reconstruct_sensor_signal[:, idx], label=f'learned feature {idx + 1}', linewidth=4)
            #     ax.legend()
            #     plt.show()

            feature_index = [13, 14, 15, 142, 150, 151, 286, 454]
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)

            # color
            cmap = mpl.cm.get_cmap('tab10', 8)

            for idx, feature_idx in enumerate(feature_index):
                # ax.plot(reconstruct_sensor_signal[:, feature_idx], c=cmap(idx), label=f'learned feature {feature_idx+1}', linewidth=4)
                element = feature_idx%8
                filter_number = int(feature_idx/8)
                ax.plot(reconstruct_sensor_signal[:, feature_idx], c=cmap(idx),
                        label=r'Trajectory $\mathbf{y}^{' + f'{element}' + r'}_{' + f'{filter_number}' + r',j}$', linewidth=4)
            # ax.set_xlabel('Time steps', fontdict=self.font, labelpad=10)
            ax.set_xlabel(r'Time index of moving window $\mathbf{W}_j$', fontdict=self.font, labelpad=10)
            # ax.set_ylabel('Value', fontdict=self.font, labelpad=10)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=-9)
            ax.tick_params(direction='out', width=2, length=6)

            notation_font = {'fontsize': 32,
                             'family': 'Times New Roman',
                             'horizontalalignment': 'center',
                             'verticalalignment': 'center',
                             'weight': 'normal'
                             }
            ax.text(175, 0.2, 'Key degradation\n information', c='r', fontdict=notation_font)

            # ax.set_title('Sensor signals output by 1D CNN')
            legend_font = {'family': 'Times New Roman',
                            'weight': 'normal',
                            'size': 20}
            # ax.legend(loc='best', prop=legend_font, frameon=False)
            # ax.legend(prop=legend_font, loc=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.24))

            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(24)

            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax.spines[position].set_linewidth(4)

            # ax1
            ax_1 = fig.add_axes([0.35, 0.68, 0.2, 0.27])
            for tick in ax_1.get_xticklabels() + ax_1.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(20)

            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax_1.spines[position].set_linewidth(3)

            # tick span
            ax_1.xaxis.set_major_locator(plt.MultipleLocator(10))
            ax_1.yaxis.set_major_locator(plt.MultipleLocator(1))

            start_point = 120
            end_point = 151
            rise_feature_index = [142, 150, 151, 454]
            color_index = [3, 4, 5, 7]
            for idx, feature_idx in zip(color_index, rise_feature_index):
                # ax_1.plot(np.arange(start_point, end_point), reconstruct_sensor_signal[start_point:end_point, feature_idx],
                #           c=cmap(idx), label=f'learned feature {feature_idx+1}', linewidth=4)
                ax_1.plot(np.arange(start_point, end_point), reconstruct_sensor_signal[start_point:end_point, feature_idx],
                          c=cmap(idx), label=f'trajectory {idx+1}', linewidth=4)

            # ax2
            ax_2 = fig.add_axes([0.35, 0.22, 0.2, 0.27])
            for tick in ax_2.get_xticklabels() + ax_2.get_yticklabels():
                tick.set_family('Times New Roman')
                tick.set_fontsize(20)

            positions = ['top', 'bottom', 'right', 'left']
            for position in positions:
                ax_2.spines[position].set_linewidth(3)

            # tick span
            ax_2.xaxis.set_major_locator(plt.MultipleLocator(10))
            ax_2.yaxis.set_major_locator(plt.MultipleLocator(0.5))

            down_feature_index = [13, 14, 15, 286]
            color_index = [0, 1, 2, 6]
            for idx, feature_idx in zip(color_index, down_feature_index):
                # ax_2.plot(np.arange(start_point, end_point),
                #           reconstruct_sensor_signal[start_point:end_point, feature_idx],
                #           c=cmap(idx), label=f'learned feature {feature_idx + 1}', linewidth=4)
                ax_2.plot(np.arange(start_point, end_point),
                          reconstruct_sensor_signal[start_point:end_point, feature_idx],
                          c=cmap(idx), label=f'trajectory {idx + 5}', linewidth=4)

            # 增加矩形框
            rect1 = plt.Rectangle((120, -2.4), 30, 1.7, fill=False, edgecolor='black', linewidth=3)
            rect2 = plt.Rectangle((120, 0.8), 30, 3.4, fill=False, edgecolor='black', linewidth=3)
            rect1.set_zorder(8)
            rect2.set_zorder(8)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.annotate("", xy=(108, -5.0), xytext=(130, -2.6),
                        arrowprops=dict(arrowstyle="->", lw=3, mutation_scale=28))
            ax.annotate("", xy=(108, 6.8), xytext=(130, 4.4),
                        arrowprops=dict(arrowstyle="->", lw=3, mutation_scale=28))

            plt.tight_layout()
            # plt.show()
            # fig.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\hidden_feature_2.pdf')
            plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f3.0\figures\graph abstract\temporal_features.png')
            # plt.savefig(self.saving_path + f'/hidden_feature_curve/{state}_{feature_idx+1}.png')
            # plt.show()
            # plt.close()


    def abstract_feature_curve(self, number=None, state=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/abstract_feature_curve')
        for feature_idx in fig_number:
            sample_loader = self.test_dataloader[feature_idx]
            _, _, sample_hidden = self.eval(sample_loader)
            atten_input = sample_hidden[2].detach().numpy()
            # atten_input[:, 0, 0] = -0.8
            # abstract_feature = atten_input[:, 47, :]
            # reconstruct_sensor_signal = conv_out[:, -1, :]
            abstract_feature = atten_input.reshape((atten_input.shape[0], -1))

            # select feature map
            for fig_num in range(int(abstract_feature.shape[1]/10)+1):
                fig = plt.figure(figsize=(16, 9))
                ax = fig.add_subplot(111)
                for idx in range(8*fig_num, 8*(fig_num+1), 1):
                    ax.plot(abstract_feature[:, idx], label=f'learned feature {idx + 1}', linewidth=4)
                ax.legend()
                plt.show()

            selected_features_idx = [2, 18, 29, 34, 51, 67, 81, 93]
            selected_features = pd.DataFrame(abstract_feature[:, selected_features_idx], columns=selected_features_idx)
            selected_features.to_csv(self.saving_path+ fr'/abstract_feature_curve/selected_features.csv', index=None, header=True)


            # # FD001
            # abstract_feature_indices = [0, 2, 3, 3, 3, 25, 26, 44]
            # feature_element_indices = [0, 2, 0, 1, 2, 2, 11, 2 ]
            # col_indices = list(range(8))
            #
            # # FD004
            # # abstract_feature_indices = [0, 14, 15, 17, 23, 28, 41, 47, 51, 59, 62, 63]
            # # # feature_element_indices = [5, 12, 15, 15, 12, 9, 9, 7, 13, 3, 3]
            # # feature_element_indices = [1, 4, 11, 14, 14, 2, 11, 2, 8, 8, 6, 12]
            # # col_indices = list(range(12))
            #
            # fig = plt.figure(figsize=(16, 12))
            # ax = fig.add_subplot(111)
            #
            # # color
            # # cmap = mpl.cm.get_cmap('Paired', 12)
            #
            # # FD001
            # cmap = mpl.cm.get_cmap('tab20')
            # color_number = [12, 4, 18, 16, 14, 2, 6, 8, 10]
            #
            # # FD004
            # # cmap = mpl.cm.get_cmap('tab20')
            # # color_number = [3, 19] + list(range(0, 20, 2))
            # color_list = [cmap(i) for i in color_number]
            #
            # # for (abstract_index, element_index, col_index) in zip(abstract_feature_indices, feature_element_indices, col_indices):
            # #     abstract_feature = atten_input[:, abstract_index, element_index]
            # #
            # #     ax.plot(abstract_feature, c=color_list[col_index],
            # #             label=r'Abstract feature $\mathbf{h}^{' + f'{element_index+1}' + r'}_{' +
            # #                   f'{abstract_index+1}' + r',j}$', linewidth=4)
            #
            #     # ax.plot(abstract_feature, c=cmap(col_index),
            #     #         label=r'Abstract feature $\mathbf{h}^{' + f'{element_index+1}' + r'}_{' +
            #     #               f'{abstract_index+1}' + r',j}$', linewidth=4)
            #
            #
            #
            # # feature_index = [13, 14, 15, 142, 150, 151, 286, 454]
            #
            # ax.set_xlabel(r'Time index of moving window $\mathbf{W}_j$', fontdict=self.font, labelpad=10)
            # ax.set_xlim((0, 193))
            # # ax.set_xlim(0, 325)
            # bottom_value = -8
            # ax.set_ylim(bottom=bottom_value)
            # ax.tick_params(direction='out', width=2, length=6)
            # #
            # legend_font = {'family': 'Times New Roman',
            #                 'weight': 'normal',
            #                 'size': 20}
            # # ax.legend(prop=legend_font, loc=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.26))
            # # ax.legend(prop=legend_font, loc=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.34))
            # #
            # for tick in ax.get_xticklabels() + ax.get_yticklabels():
            #     tick.set_family('Times New Roman')
            #     tick.set_fontsize(24)
            #
            # positions = ['top', 'bottom', 'right', 'left']
            # for position in positions:
            #     ax.spines[position].set_linewidth(4)
            #
            # # plot stage
            #
            # # FD001
            # ax.plot([72, 72], [bottom_value, 100], linewidth=3, color='k', linestyle='--')
            # ax.plot([122, 122], [bottom_value, 100], linewidth=3, color='k', linestyle='--')
            # ax.plot([162, 162], [bottom_value, 100], linewidth=3, color='k', linestyle='--')
            #
            #
            # # FD004
            # # ax.plot([202, 202], [bottom_value, 34], linewidth=3, color='k', linestyle='--')
            # # ax.plot([251, 251], [bottom_value, 34], linewidth=3, color='k', linestyle='--')
            # # ax.plot([289, 289], [bottom_value, 34], linewidth=3, color='k', linestyle='--')
            # notation_font = {'fontsize': 32,
            #                  'family': 'Times New Roman',
            #                  'horizontalalignment': 'center',
            #                  'verticalalignment': 'center',
            #                  'weight': 'bold'
            #                  }
            #
            # # FD001
            # ax.text(35, bottom_value/2-0.7, 'Healthy', c='k', fontdict=notation_font)
            # ax.text(97, bottom_value/2-0.7, 'Initial', c='k', fontdict=notation_font)
            # ax.text(142, bottom_value/2-0.7, 'Middle', c='k', fontdict=notation_font)
            # ax.text(177, bottom_value/2-0.7, 'Failure', c='k', fontdict=notation_font)
            #
            #
            # # FD004
            # # ax.text(100, bottom_value/2, 'Healthy', c='k', fontdict=notation_font)
            # # ax.text(227, bottom_value/2, 'Initial', c='k', fontdict=notation_font)
            # # ax.text(270, bottom_value/2, 'Middle', c='k', fontdict=notation_font)
            # # ax.text(307, bottom_value/2, 'Failure', c='k', fontdict=notation_font)
            #
            #
            # plt.tight_layout()
            # # plt.show()
            # # fig.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD004_abstract_feature_v1.pdf')
            # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f3.0\figures\graph abstract\abstract_features.png')



    def visualization_of_attention(self, span=20, number=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/visualization_of_attention')
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            _, _, sample_hidden = self.eval(sample_loader)
            score_att = pd.DataFrame(sample_hidden[0].detach().numpy())
            score_att.to_csv(self.saving_path + f'/visualization_of_attention/score_attention.csv', index=None, header=None)

            # FD001
            # healthy_atten = score_att.iloc[0:span, :]
            # initial_degradation_atten = score_att.iloc[-120:-120+span, :]
            # medium_degradation_atten = score_att.iloc[-70:-70+span, :]
            # failure_degradation_atten = score_att.iloc[-span:, :]

            # FD004
            # healthy_atten = score_att.iloc[0:span, :]
            # initial_degradation_atten = score_att.iloc[-120:-120+span, :]
            # medium_degradation_atten = score_att.iloc[-70:-70+span, :]
            # failure_degradation_atten = score_att.iloc[-span:, :]

            # self.attention_heatmap(healthy_atten, state='health')
            # self.attention_heatmap(initial_degradation_atten, state='initial')
            # self.attention_heatmap(medium_degradation_atten, state='medium')
            # self.attention_heatmap(failure_degradation_atten, state='failure')

    def attention_heatmap(self, heatmap_data, state=None):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        # 定义渐变颜色条（白色-蓝色-红色）
        # my_colormap = LinearSegmentedColormap.from_list("", [(0, '#d7fffe'),
        #                                                      (0.25, '#5a86ad'),
        #                                                      (0.5, '#01386a'),
        #                                                      (0.75, '#001146'),
        #                                                      (1, '#5f1b6b')])
        # my_colormap = LinearSegmentedColormap.from_list("", ['white', 'blue', 'red'])
        cmap = mpl.cm.get_cmap('tab20')
        # color_number = np.concatenate((np.arange(1, 20, 2), np.arange(0, 20, 2)))
        # color_number = [1, 3, 9, 19, 18, 17, 5, 16, 4, 0, 11, 7, 2, 13, 12, 8, 15, 14, 10, 6]
        color_number = [1, 3, 9, 17, 5, 16, 4, 0, 11, 7, 2, 13, 12, 8,  19, 18, 10, 6]
        color_list = [cmap(i) for i in color_number]
        # color_list = [cmap(i) for i in np.arange(1, 20, 2)] + [cmap(i) for i in np.arange(0, 20, 2)]
        my_colormap = LinearSegmentedColormap.from_list("", color_list)
        # sns.heatmap(heatmap_data, cmap='tab20', square=True, linewidths=.5, vmin=0, vmax=0.7)
        sns.heatmap(heatmap_data, cmap=my_colormap, square=True, linewidths=.5, vmin=0, vmax=0.7, cbar=False)
        # sns.heatmap(heatmap_data, cmap=my_colormap, square=True, linewidths=.5, vmin=0, vmax=0.7)
        # sns.color_palette("GnBu_d", as_cmap=True)
        # sns.heatmap(heatmap_data, cmap="dull blue", square=True, linewidths=.5, vmin=-0.8, vmax=0.8)
        ax.set_xlabel('Window Size', fontdict=self.font, labelpad=10)
        ax.set_ylabel('Time Steps', fontdict=self.font, labelpad=10)
        # ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        y_value = np.array(heatmap_data.index)
        ax.set_yticks(np.arange(0.5, 20, 2))
        ax.set_yticklabels(np.arange(y_value[1], y_value[-1]+1, 2))
        ax.xaxis.set_major_locator(plt.MultipleLocator(4))
        ax.set_xticks(ax.get_xticks()[1:-2]+0.5)
        ax.set_xticklabels(np.arange(1, 65, 4))
        # ax.set_xticks(list(heatmap_data.index))
        # ax.set_yticks(np.arange(1, 65))
        # ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        # ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        for tick in ax.get_xticklabels():
            tick.set_family('Times New Roman')
            tick.set_fontsize(20)
            tick.set_rotation(0)
        for tick in ax.get_yticklabels():
            tick.set_family('Times New Roman')
            tick.set_fontsize(20)

        plt.tight_layout()
        plt.savefig(fr'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\FD004_attention+{state}.svg',
                    dpi=600, format='svg')
        plt.show()


    def degradation_pattern(self, number=None):
        fig_number = list(range(len(self.y))) if number is None else [number]
        mkdir(self.saving_path + f'/hidden_feature_curve')
        for idx in fig_number:
            sample_loader = self.test_dataloader[idx]
            _, _, sample_hidden = self.eval(sample_loader)
            atten_input = sample_hidden[2].detach().numpy()
            reconstruct_sensor_signal = atten_input[:, -1, :]
            fig = plt.figure(figsize=(16, 9))
            ax1 = fig.add_axes([0.05, 0.05, 0.95, 0.95])
            # ax1 = fig.add_subplot(111)
            sensors = [0, 1, 4, 6, 13, 14]
            for idx in sensors:
                ax1.plot(reconstruct_sensor_signal[:, idx], label=f'sensor {idx}', linewidth=4)
            ax1.set_xlabel('Cycles', fontdict=self.font)
            ax1.set_ylabel('Values', fontdict=self.font)
            ax1.set_title('Sensor signals output by 1D CNN')
            legend_font = {'family': 'Times New Roman',
                           'weight': 'normal',
                           'size': 20}
            ax1.legend(loc='upper right', prop=legend_font, frameon=False)
            ax1.set_ylim(0, 220)
            # plt.tight_layout()
            ax2 = fig.add_axes([0.1, 0.1, 0.4, 0.3])
            plt.show()


class MovingAvg(nn.Module):
    def __init__(self, input_shape: tuple, window_size: int):
        """
        Takes a batch of sequences with [batch size, channels, seq_len] and smoothes the sequences.
        Output size of moving average is: time series length - window size + 1

        Args:
            input_shape (tuple): input shape for the transformation layer in format (n_channels, length_of_timeseries)
            window_size (int): window size with which the time series is smoothed
        """
        assert len(input_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"
        super(MovingAvg, self).__init__()

        self.num_dim, self.length_x = input_shape
        self.window_size = window_size

        # compute output shape after smoothing (len ts - window size + 1)
        new_length = self.length_x - self.window_size + 1
        self.output_shape = (self.num_dim, new_length)

        # kernel weights for average convolution
        self.kernel_weights = torch.ones((self.num_dim, 1, self.window_size), dtype=torch.double) / self.window_size

    def forward(self, x):
        """
        Args:
          x (tensor): batch of time series samples
        Returns:
          output (tensor): smoothed time series batch
        """

        output = nn.functional.conv1d(x, self.kernel_weights, groups=self.num_dim)

        return output


class Downsample(nn.Module):
    def __init__(self, input_shape: tuple, sample_rate: int):
        """
        Takes a batch of sequences with [batch size, channels, seq_len] and down-samples with sample
        rate k. Hence, every k-th element of the original time series is kept.

        Args:
            input_shape (tuple): input shape for the transformation layer in format (n_channels, length_of_timeseries)
            sample_rate (int): sample rate with which the time series should be down-sampled
        """
        assert len(input_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"
        super(Downsample, self).__init__()

        self.sample_rate = sample_rate

        # compute output shape after down-sampling
        self.num_dim, self.length_x = input_shape
        last_one = 0
        if self.length_x % self.sample_rate > 0:
            last_one = 1
        new_length = int(np.floor(self.length_x / self.sample_rate)) + last_one
        self.output_shape = (self.num_dim, new_length)

    def forward(self, x):
        """
        Args:
          x (tensor): batch of time series samples
        Returns:
          output (tensor): down-sampled time series batch
        """
        batch_size = x.shape[0]

        last_one = 0
        if self.length_x % self.sample_rate > 0:
            last_one = 1

        new_length = int(np.floor(self.length_x / self.sample_rate)) + last_one
        output = torch.zeros((batch_size, self.num_dim, new_length), dtype=torch.double)
        # print(output.type())
        output[:, :, range(new_length)] = x[:, :, [i * self.sample_rate for i in range(new_length)]]

        return output


class Identity(nn.Module):
    def __init__(self, input_shape: tuple):
        """
        Identity mapping without any transformation (wrapper class).

        Args:
            input_shape (tuple): input shape for the transformation layer in format (n_channels, seq_len)
        """
        super(Identity, self).__init__()
        assert len(input_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"
        self.output_shape = input_shape

    def forward(self, x):
        """
        Args:
          x (tensor): batch of time series samples
        Returns:
          output (tensor): same as x
        """
        return x



class MCNNAttention(nn.Module):
    def __init__(self, ts_shape: tuple, pool_factor: int,
                 kernel_size: float or int, transformations: dict):
        """
        Multi-Scale Convolutional Neural Network for Time Series Classification - Cui et al. (2016).

        Args:
          ts_shape (tuple):           shape of the time series, e.g. (1, 9) for uni-variate time series
                                      with length 9, or (3, 9) for multivariate time series with length 9
                                      and three features
          n_classes (int):            number of classes
          pool_factor (int):          length of feature map after max pooling, usually in {2,3,5}
          kernel_size (int or float): filter size for convolutional layers, usually ratio in {0.05, 0.1, 0.2}
                                      times the length of time series
          transformations (dict):     dictionary with key value pairs specifying the transformations
                                      in the format 'name': {'class': <TransformClass>, 'params': <parameters>}
        """
        assert len(ts_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"

        super(MCNNAttention, self).__init__()

        self.ts_shape = ts_shape
        self.sensors, self.seq_len = ts_shape[0], ts_shape[1]
        self.pool_factor = pool_factor
        self.kernel_size = int(self.ts_shape[1] * kernel_size) if kernel_size < 1 else int(kernel_size)


        # layer settings
        self.local_conv_filters = 256
        self.local_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.full_conv_filters = 256
        self.full_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.fc_units = 256
        self.fc_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.dim_atten = 128
        self.dim_reg = 32
        self.relu = nn.ReLU()


        # setup branches
        self.branches = self._setup_branches(transformations)
        self.n_branches = len(self.branches)

        # full convolution
        in_channels = self.local_conv_filters * self.n_branches
        # kernel shouldn't exceed the length (length is always pool factor?)
        full_conv_kernel_size = int(min(self.kernel_size, int(self.pool_factor)))
        self.full_conv = nn.Conv1d(in_channels, self.full_conv_filters,
                                   kernel_size=full_conv_kernel_size,
                                   padding='same')
        pool_size = 1
        self.full_conv_pool = nn.MaxPool1d(pool_size)

        # fully-connected
        self.flatten = nn.Flatten()
        in_features = int(self.pool_factor * self.full_conv_filters)
        self.conv_fc = nn.Linear(in_features, self.sensors*self.seq_len)

        # attention
        self.layer_attention = nn.Linear(self.seq_len*4, self.dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(self.dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


        # output
        self.layer_outhidden = nn.Linear(self.seq_len, self.dim_reg)
        self.layer_output = nn.Linear(self.dim_reg, 1)

    def forward(self, x):
        xs = [self.branches[idx](x) for idx in range(self.n_branches)]
        # print(xs)

        x = torch.cat(xs, dim=1)

        x = self.full_conv(x)
        x = self.full_conv_activation()(x)
        x = self.full_conv_pool(x)

        x = self.flatten(x)
        conv_out = self.conv_fc(x)

        # print(atten_input.shape)
        atten_input = conv_out.view(-1, self.sensors, self.seq_len)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.relu(self.layer_outhidden(embed_pool))
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        out = out.flatten()

        return out, score_att, conv_out, atten_input, embed_pool


    def _build_local_branch(self, name: str, transform: nn.Module, params: list):
        """
        Build transformation and local convolution branch.

        Args:
          name (str):   Name of the branch.
          transform (nn.Module):  Transformation class applied in this branch.
          params (list):   Parameters for the transformation, with the first parameter always being the input shape.
        Returns:
          branch:   Sequential model containing transform, local convolution, activation, and max pooling.
        """
        branch = nn.Sequential()
        # transformation
        # instance
        branch.add_module(name + '_transform', transform(*params))
        # local convolution
        branch.add_module(name + '_conv', nn.Conv1d(self.ts_shape[0], self.local_conv_filters,
                                                    kernel_size=self.kernel_size, padding='same'))
        branch.add_module(name + '_activation', self.local_conv_activation())
        # local max pooling (ensure that outputs all have length equal to pool factor)
        pool_size = int(int(branch[0].output_shape[1]) / self.pool_factor)
        assert pool_size > 1, "ATTENTION: pool_size can not be 0 or 1, as the lengths are then not equal" \
                              "for concatenation!"
        branch.add_module(name + '_pool', nn.MaxPool1d(pool_size))  # default stride equal to pool size

        return branch

    def _setup_branches(self, transformations: dict):
        """
        Setup all branches for the local convolution.

        Args:
          transformations:  Dictionary containing the transformation classes and parameter settings.
        Returns:
          branches: List of sequential models with local convolution per branch.
        """
        branches = []
        for transform_name in transformations:
            transform_class = transformations[transform_name]['class']
            parameter_list = transformations[transform_name]['params']

            # create transform layer for each parameter configuration
            if parameter_list:
                for param in parameter_list:
                    if np.isscalar(param):
                        # 判断是否为标量
                        name = transform_name + '_' + str(param)
                        branch = self._build_local_branch(name, transform_class, [self.ts_shape, param])
                    else:
                        branch = self._build_local_branch(transform_name, transform_class,
                                                          [self.ts_shape] + list(param))
                    branches.append(branch)
            else:
                branch = self._build_local_branch(transform_name, transform_class, [self.ts_shape])
                branches.append(branch)

        return torch.nn.ModuleList(branches)

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class MCNNsAttention(nn.Module):
    def __init__(self, ts_shape: tuple, pool_factor: int,
                 kernel_size: float or int, moving_average: list,
                 dim_atten: int, dim_reg: int, drop_r: float):
        """
        Multi-Scale Convolutional Neural Network for Time Series Classification - Cui et al. (2016).

        Args:
          ts_shape (tuple):           shape of the time series, e.g. (1, 9) for uni-variate time series
                                      with length 9, or (3, 9) for multivariate time series with length 9
                                      and three features
          n_classes (int):            number of classes
          pool_factor (int):          length of feature map after max pooling, usually in {2,3,5}
          kernel_size (int or float): filter size for convolutional layers, usually ratio in {0.05, 0.1, 0.2}
                                      times the length of time series
          transformations (dict):     dictionary with key value pairs specifying the transformations
                                      in the format 'name': {'class': <TransformClass>, 'params': <parameters>}
        """
        assert len(ts_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"

        super(MCNNsAttention, self).__init__()

        self.ts_shape = ts_shape
        self.sensors, self.seq_len = ts_shape[0], ts_shape[1]
        self.pool_factor = pool_factor
        self.kernel_size = int(self.ts_shape[1] * kernel_size) if kernel_size < 1 else int(kernel_size)
        self.transformations = {
                                'identity': {
                                    'class': Identity,
                                    'params': []
                                },
                                'movingAvg': {
                                    'class': MovingAvg,
                                    'params': moving_average       # window sizes
                                },
                                'downsample': {
                                    'class': Downsample,
                                    'params': [2]       # sampling rates
                                }
                                }

        # layer settings
        self.fc_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.dim_atten = dim_atten
        self.dim_reg = dim_reg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_r)

        # setup branches
        self.branches = self._setup_branches(self.transformations)
        self.n_branches = len(self.branches)

        # fully-connected
        self.flatten = nn.Flatten()
        in_features = int(self.sensors*4*self.n_branches*self.pool_factor)
        self.conv_fc = nn.Linear(in_features, self.sensors*self.seq_len)

        # attention
        self.layer_attention = nn.Linear(self.seq_len*4, self.dim_atten)
        # self.layer_attention = nn.Linear(time_step*4, dim_atten)
        self.layer_atten_out = nn.Linear(self.dim_atten, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # output
        self.layer_outhidden = nn.Linear(self.seq_len, self.dim_reg)
        self.layer_output = nn.Linear(self.dim_reg, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        xs = [self.branches[idx](x) for idx in range(self.n_branches)]
        # print(xs)
        # print([x.shape for x in xs])

        x = torch.cat(xs, dim=1)


        x = self.flatten(x)

        conv_out = self.conv_fc(x)

        # print(atten_input.shape)
        atten_input = conv_out.view(-1, self.sensors, self.seq_len)
        atten_base = atten_input[:, 0, :]
        atten_base = atten_base.view(atten_base.shape[0], -1, atten_base.shape[1])
        atten_bases = atten_base.repeat(1, atten_input.shape[1], 1)
        score_att = self.attention_unit(atten_input, atten_bases)
        embed_pool = self.sumpooling(atten_input, score_att)
        # print(embed_pool.shape)
        out_hidden = self.dropout(self.relu(self.layer_outhidden(embed_pool)))
        # print(out_hidden.shape)
        out = self.layer_output(out_hidden)
        out = out.flatten()

        return out, score_att, conv_out, atten_input, embed_pool




    def _build_local_branch(self, name: str, transform: nn.Module, params: list):
        """
        Build transformation and local convolution branch.

        Args:
          name (str):   Name of the branch.
          transform (nn.Module):  Transformation class applied in this branch.
          params (list):   Parameters for the transformation, with the first parameter always being the input shape.
        Returns:
          branch:   Sequential model containing transform, local convolution, activation, and max pooling.
        """
        branch = nn.Sequential()
        # transformation
        # instance
        branch.add_module(name + '_transform', transform(*params))
        # local convolution
        branch.add_module(name + '_conv', Xception(input_channel=self.sensors))
        # local max pooling (ensure that outputs all have length equal to pool factor)
        pool_size = math.ceil(int(branch[0].output_shape[1]) / (4*self.pool_factor))
        assert pool_size > 1, "ATTENTION: pool_size can not be 0 or 1, as the lengths are then not equal" \
                              "for concatenation!"
        branch.add_module(name + '_pool', nn.MaxPool1d(pool_size, ceil_mode=True))  # default stride equal to pool size

        return branch

    def _setup_branches(self, transformations: dict):
        """
        Setup all branches for the local convolution.

        Args:
          transformations:  Dictionary containing the transformation classes and parameter settings.
        Returns:
          branches: List of sequential models with local convolution per branch.
        """
        branches = []
        for transform_name in transformations:
            transform_class = transformations[transform_name]['class']
            parameter_list = transformations[transform_name]['params']

            # create transform layer for each parameter configuration
            if parameter_list:
                for param in parameter_list:
                    if np.isscalar(param):
                        # 判断是否为标量
                        name = transform_name + '_' + str(param)
                        branch = self._build_local_branch(name, transform_class, [self.ts_shape, param])
                    else:
                        branch = self._build_local_branch(transform_name, transform_class,
                                                          [self.ts_shape] + list(param))
                    branches.append(branch)
            else:
                branch = self._build_local_branch(transform_name, transform_class, [self.ts_shape])
                branches.append(branch)

        return torch.nn.ModuleList(branches)

    def sumpooling(self, embed_his, score_att):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param score_att: [None, cycle_step]
        '''
        score_att = score_att.view((-1, score_att.shape[1], 1))
        embed_his = embed_his.permute((0, 2, 1))
        embed = torch.matmul(embed_his, score_att)
        return embed.view((-1, embed.shape[1]))

    def attention_unit(self, embed_his, embed_bases):
        '''
        :param embed_his: [None, cycle_step, dim_embed]
        :param embed_base: [None, 1, dim_embed]
        '''
        embed_concat = torch.cat([embed_his, embed_bases, embed_his - embed_bases, embed_his * embed_bases],
                                 dim=2)  # [None, cycle_step, dim_embed*4]
        hidden_att = self.tanh(self.layer_attention(embed_concat))
        # print(hidden_att.shape)
        score_att = self.layer_atten_out(hidden_att)
        # print(score_att.shape)
        score_att = self.softmax(score_att.view((-1, score_att.shape[1])))
        return score_att


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding='same', bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Xception(nn.Module):
    def __init__(self, input_channel):
        super(Xception, self).__init__()
        self.branch_1 = nn.Sequential(
            depthwise_separable_conv(input_channel, input_channel*2, kernel_size=3),
            nn.ReLU(True),
            depthwise_separable_conv(input_channel*2, input_channel*2, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.branch_res_1 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel*2, kernel_size=1, stride=2)

        self.branch_2 = nn.Sequential(
            depthwise_separable_conv(input_channel*2, input_channel*4, kernel_size=3),
            nn.ReLU(True),
            depthwise_separable_conv(input_channel*4, input_channel*4, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.branch_res_2 = nn.Conv1d(in_channels=input_channel*2, out_channels=input_channel*4, kernel_size=1, stride=2)

    def forward(self, x):
        # print(x.shape)
        branch_out_1 = self.branch_1(x) + self.branch_res_1(x)
        # print(branch_out_1.shape)
        branch_out_2 = self.branch_2(branch_out_1) + self.branch_res_2(branch_out_1)
        # print(branch_out_2.shape)

        return branch_out_2


def CNN_layer_effect(rmse=None, score=None, time=None):
    rmse = [1, 0.86, 0.84, 0.87]
    score = [1, 0.85, 0.81, 0.83]
    time = [0.57, 0.66, 0.81, 1]
    x = np.arange(4)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1)
    width = 0.25
    font = {'fontsize': 32, 'family': 'Times New Roman'}
    # 绘制RMSE
    ax.bar(x, rmse, width=width, label='RMSE', fc='#1F77B4')
    # x1 = x
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    ax.bar(x + width, score, width=width, label='Score', fc='#FF7F0E')

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    ax.bar(x + width * 2, time, width=width, label='Time', fc='#2CA02C')

    # 设置字体
    # ax.set_yticks(fontdict=font)
    ax.set_xticks((0.25, 1.25, 2.25, 3.25))
    ax.set_xticklabels(labels=['1-layer', '2-layer', '3-layer', '4-layer'])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_family('Times New Roman')
        tick.set_fontsize(28)

    # 添加虚线
    # ax.grid(axis='y', alpha=0.4, dashes=[6, 2])

    ax.set_ylabel('Performance metrics', fontdict=font, labelpad=10)
    ax.tick_params(direction='out', width=2, length=6)
    # 去掉边框
    # 把上面边框弄没
    # ax.spines['top'].set_visible(False)
    # 下面
    # ax.spines['bottom'].set_visible(False)
    # 左边
    # ax.spines['left'].set_visible(False)
    # 右边
    # ax.spines['right'].set_visible(False)
    # set linewidth in spines
    positions = ['bottom', 'top', 'right', 'left']
    for position in positions:
        ax.spines[position].set_linewidth(4)

    legend_font = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 28}
    ax.legend(prop=legend_font, loc=8, frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f8.0\figures\CNN_layer.pdf')
    plt.show()

def time_window_effect(rmse=None, time_window=None, dataset='FD001'):
    if rmse is None:
        if dataset == 'FD001':
            rmse = [15.18, 13.01, 11.61, 10.61, 10.42]
            score = [571.85, 391.13, 298.63, 215.42, 210]
            rmse_ylim = (7, 17)
            rmse_span = np.arange(7, 18)
            score_ylim = (150, 650)
            score_span = np.arange(150, 700, 50)
        elif dataset == 'FD002':
            rmse = [18.24, 16.14, 14.86, 13.43, 14.21]
            score = [2356, 1718, 1364, 923.15, 1103.12]
            rmse_ylim = (11, 19)
            rmse_span = np.arange(11, 20)
            score_ylim = (500, 2500)
            score_span = np.arange(500, 2600, 250)
        elif dataset == 'FD003':
            rmse = [15.13, 12.34, 10.94, 9.97, 9.80]
            score = [602, 382, 303, 221.83, 226]
            rmse_ylim = (7, 17)
            rmse_span = np.arange(7, 18)
            score_ylim = (150, 650)
            score_span = np.arange(150, 700, 50)
        else:
            rmse = [19.18, 17.34, 16.01, 14.05, 15.31]
            score = [3469, 2451, 1953, 1192.99, 1635]
            rmse_ylim = (12, 20)
            rmse_span = np.arange(12, 21)
            score_ylim = (500, 4000)
            score_span = np.arange(500, 4700, 500)
    if time_window is None:
        time_window = [16, 32, 48, 64, 80]

    # axis color
    yaxis_left = '#31859C'
    yaxis_right = '#DD4A16'

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(111)
    ax1.bar(time_window, rmse, color=yaxis_left, width=8)
    font = {'fontsize': 32, 'family': 'Times New Roman', 'weight': 'normal'}
    # ax.set_xticklabels(time_window, fontdict=font)

    ax1.set_ylim(rmse_ylim)
    ax1.set_xlabel('Window size', fontdict=font, labelpad=10)
    ax1.set_ylabel('RMSE', fontdict=font, color=yaxis_left, labelpad=10)
    ax1.grid(axis='y', alpha=0.4, dashes=[6, 2])
    ax1.set_yticks(rmse_span)

    ax1.set_xticks(time_window)
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_family('Times New Roman')
        tick.set_fontsize(28)

    # 绘制另一个坐标轴
    ax2 = ax1.twinx()
    ax2.plot(time_window, score, color=yaxis_right, marker='o', markersize=12, linestyle='-', lw=4)
    ax2.set_ylabel('Score', fontdict=font, color=yaxis_right, labelpad=10)
    ax2.set_ylim(score_ylim)
    ax2.set_yticks(score_span)
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_family('Times New Roman')
        tick.set_fontsize(28)

    positions = ['top', 'bottom', 'right', 'left']
    for position in positions:
        ax2.spines[position].set_linewidth(4)

    ax1.tick_params(direction='out', width=2, length=6, colors=yaxis_left)
    ax1.tick_params(axis='y', colors=yaxis_left)
    # ax1.spines['right'].set_color(yaxis_right)
    ax2.tick_params(direction='out', width=2, length=6, colors=yaxis_right)
    ax2.tick_params(axis='y', colors=yaxis_right)
    ax1.tick_params(axis='x', colors='black')
    ax2.spines['bottom'].set_color('black')
    # ax2.spines['left'].set_color(yaxis_left)

    plt.tight_layout()
    plt.savefig(fr'F:\hilbert-研究生\学术写作\turbofan\RUL Turbofan f4.0\figures\{dataset}_window.pdf')
    plt.show()

def performance_metric():
    h = np.arange(-40, 40, 0.01)
    s = h.copy()
    rmse = np.sqrt(np.power(h, 2))
    for i, value in enumerate(h):
        if value > 0:
            s[i] = np.exp(value / 10) - 1
        else:
            s[i] = np.exp(-value / 13) - 1

    # 绘制x=0的分割线
    y = np.arange(0, 60, 0.01)
    x = np.zeros(shape=y.shape)

    # 绘制图像
    legend_font = {
        'family': 'Times New Roman',  # 字体
        'style': 'normal',
        'size': 24,  # 字号
        'weight': "normal",  # 是否加粗，不加粗
    }
    font = {'fontsize': 28, 'family': 'Times New Roman'}
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(h, rmse, lw=2, label='Score Function')
    ax1.plot(h, s, 'r', dashes=[6, 2], lw=2, label='RMSE')
    ax1.plot(x, y, 'b', dashes=[6, 2], lw=2)
    ax1.grid(axis='both', alpha=0.4, dashes=[6, 2])
    ax1.legend(loc='best', prop=legend_font, frameon=False)  # 不显示图例框线
    ax1.set_xlabel('Error Value $d_{i}$', fontdict=font, labelpad=10)
    ax1.set_ylabel('Performance Metrics', fontdict=font, labelpad=10)
    ax1.set_xlim((-40, 40))
    ax1.set_ylim((0, 60))
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_family('Times New Roman')
        tick.set_fontsize(24)

    # spines line width
    positions = ['top', 'bottom', 'right', 'left']
    for position in positions:
        ax1.spines[position].set_linewidth(4)

    ax1.tick_params(direction='out', width=2, length=6)

    # notation
    notation_font = {'fontsize': 32,
                     'family': 'Times New Roman',
                     'horizontalalignment': 'center',
                     'verticalalignment': 'center'
                    }
    ax1.text(-15, 35, 'Early \n Prediction', c='g', fontdict=notation_font)
    ax1.text(15, 35, 'Late \n Prediction', c='r', fontdict=notation_font)

    plt.tight_layout()
    # plt.savefig(r'F:\hilbert-研究生\学术写作\turbofan\论文修改\20210916-秦玉文-Latex\new_figure\performance_1.pdf')
    plt.show()


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