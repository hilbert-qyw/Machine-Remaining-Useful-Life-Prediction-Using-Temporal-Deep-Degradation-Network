# coding = utf-8
"""
作者   : Hilbert
时间   :2021/11/26 16:35
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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通过paramiko从远处服务器下载文件资源到本地
author: gxcuizy
time: 2018-08-01


"""

import paramiko
import os
from stat import S_ISDIR as isdir


def down_from_remote(sftp_obj, remote_dir_name, local_dir_name):
    """远程下载文件"""
    remote_file = sftp_obj.stat(remote_dir_name)
    if isdir(remote_file.st_mode):
        # 文件夹，不能直接下载，需要继续循环
        check_local_dir(local_dir_name)
        print('开始下载文件夹：' + remote_dir_name)
        for remote_file_name in sftp.listdir(remote_dir_name):
            sub_remote = os.path.join(remote_dir_name, remote_file_name)
            sub_remote = sub_remote.replace('\\', '/')
            sub_local = os.path.join(local_dir_name, remote_file_name)
            sub_local = sub_local.replace('\\', '/')
            down_from_remote(sftp_obj, sub_remote, sub_local)
    else:
        # 文件，直接下载
        print('开始下载文件：' + remote_dir_name)
        sftp.get(remote_dir_name, local_dir_name)


def check_local_dir(local_dir_name):
    """本地文件夹是否存在，不存在则创建"""
    if not os.path.exists(local_dir_name):
        os.makedirs(local_dir_name)


if __name__ == "__main__":
    """程序主入口"""
    # 服务器连接信息
    host_name = '219.245.39.74'
    user_name = 'yuwen_hilbert'
    password = 'Qyw@1999'
    port = 22
    # 远程文件路径（需要绝对路径）
    remote_dir = '/public/home/yuwen_hilbert/Program/temporal_deep_remaining_life/turbofan_result/FD001/' \
                 'cycle64_sensor16_batchsize16_epochs200_lr0.0001_maxlife1_asymmetric_times1'
    # 本地文件存放路径（绝对路径或者相对路径都可以）
    local_dir = '../turbofan_result'

    # # 实例化SSHClient
    # client = paramiko.SSHClient()
    #
    # # 自动添加策略，保存服务器的主机名和密钥信息，如果不添加，那么不再本地know_hosts文件中记录的主机将无法连接
    # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #
    # # 连接SSH服务端，以用户名和密码进行认证
    # client.connect(hostname=host_name, port=port, username=user_name, password=password)
    #
    # # 打开一个Channel并执行命令
    # stdin, stdout, stderr = client.exec_command('df -h ')  # stdout 为正确输出，stderr为错误输出，同时是有1个变量有值
    #
    # # 打印执行结果
    # print(stdout.read().decode('utf-8'))
    #
    # # 关闭SSHClient
    # client.close()



    # 连接远程服务器
    t = paramiko.Transport((host_name, port))
    t.connect(username=user_name, password=password)
    sftp = paramiko.SFTPClient.from_transport(t)

    # 远程文件开始下载
    down_from_remote(sftp, remote_dir, local_dir)

    # 关闭连接
    t.close()