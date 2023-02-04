# coding = utf-8
"""
作者   : Hilbert
时间   :2021/10/14 22:14
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


from lib.raw_data_processing import *

# generate processed data
# input model
CMAPSS = get_CMAPSSData(
    file_path='./CMAPSSData/raw_data/',
    outputdir='./CMAPSSData/statistical_data/',
    save_training_data=True,
    save_testing_data=True,
    files=[1, 2, 3, 4],
    MAXLIFE=130,
    min_max_norm='symmetric'
)