#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# In[1]:


get_ipython().system('pip install gym')


# In[ ]:


#1. 采用深度学习、启发式搜索、进化算法、强化学习中的任一种或两种，解决gym的cartpole-v0问题。
#2. 编写实验报告（模版在work中），说明自己实现的算法的基本原理、代码分析和性能分析。
#3. 实验报告以附件的形式提交，文件名命名规则为"学号姓名算法与程序设计大作业.doc"
#4. 作业提交截止时间为第16周周五


# In[ ]:


# 为了便于同学们修改，下面同时给出cartpole的源代码，如果需要修改reward，可在以下代码基础上进行修改


# In[11]:


# -*- coding: utf-8 -*-
# Platform: Linux python3.6.8 gcc8.3
# Linux Distribution: Linux Mint 19
# tensorflow-gpu==1.15.0, gym
# Author: Mijazz_Chan 2017326603075
# Date: May 19, 2020  01:37 AM

import random

import gym
import numpy as np
import tensorflow as tf
from numpy import mean
import tflearn
from tflearn import input_data, fully_connected, regression, DNN, dropout

# 超参

# 学习速率
learning_rate = 0.0011
env = gym.make('CartPole-v0')
env.reset()
# 学习所需的步数(random时一般都被done flag 退出, 不需过于在意)
steps_for_learn = 500
# 所取的期望分值, 高于分值即为有效策略
satisfied_score = 80
# 学习所需的游戏次数(基准learn次数, 根据gym repo, 该值越低越好)
games_for_learn = 20000
epoch = 5

# 训练数据或全局数据


# def showgame():
#     for _ in range(20):
#         env.reset()
#         # 显示效果而已, 平均分数一般都不会高于100, 所以给100步都足够
#         for _ in range(200):
#             env.render()
#             # 参照官网上写法, 具体这个sample()提供的随机值到底比random包的有什么不一样没去考究
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             if done:
#                 break
#
# showgame()

env.reset()


def get_data_from_random_game():
    # 训练数据, 含[observation, action]
    train_data = []
    # 训练的统计成绩
    scores = []
    # 超出阈值的成绩
    good_scores = []

    for _ in range(games_for_learn):
        score = 0
        # 存放每次游戏的observation和action
        local_game_result = []
        last_observation = []
        for _ in range(steps_for_learn):
            # 生产随机数0,1执行游戏, 下面再对高于satisfied的分数进行处理
            action = random.randint(0, 1)
            observation, reward, done, info = env.step(action)
            if len(last_observation) > 0:
                local_game_result.append([last_observation, action])
            last_observation = observation
            score += reward
            if done:
                break

        if score >= satisfied_score:
            good_scores.append(score)
            for result in local_game_result:
                # One-hot 数据编码预处理
                if result[1] == 1:
                    train_y = [0, 1]
                else:
                    train_y = [1, 0]
                # Training data = [train_x, train_y]
                # train_x = [observation], train_y = one-hot result
                train_data.append([result[0], train_y])

        # 一次循环下来后统计好数据跳出并继续下一次游戏环境
        scores.append(score)
        env.reset()

    print('Random Games Score Avg -->', sum(good_scores) / len(good_scores))

    return train_data


train_data = get_data_from_random_game()
# each_game[0] -> [observation]
# each_game[1] -> one-hot encoded data output
train_X = np.array([each_game[0] for each_game in train_data]).reshape(-1, len(train_data[0][0]), 1)
train_Y = [each_game[1] for each_game in train_data]

# 主模型采用tflearn的API, 构建快
# 注: dropout的rate为KeepRate, 与keras不同
network = input_data(shape=[None, len(train_X[0]), 1], name='input')

network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(network)
model.fit(train_X, train_Y, n_epoch=epoch, snapshot_step=1000, show_metric=True)

test_scores = []
actions = []

# 测试
for _ in range(100):
    score = 0
    env.reset()
    # 测试第一步给random, 根据observation来predict action
    last_observation, reward, done, info = env.step(random.randint(0, 1))
    for _ in range(steps_for_learn):
        action = np.argmax(model.predict(last_observation.reshape(-1, len(last_observation), 1))[0])
        actions.append(action)
        observation, reward, done, info = env.step(action)
        last_observation = observation
        score += reward
        if done:
            break

    test_scores.append(score)

resl = sum(test_scores) / len(test_scores)
print('Predicted Games Score Avg -->', resl)
# if resl > 199:
#     model.save('200.model')


# ```
# /home/mijazz/pyProject/tfenv/bin/python /home/mijazz/pyProject/openai-gym/gym-tf.py
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/summarizer.py:9: The name tf.summary.merge is deprecated. Please use tf.compat.v1.summary.merge instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:25: The name tf.summary.FileWriter is deprecated. Please use tf.compat.v1.summary.FileWriter instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/collections.py:13: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/config.py:123: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/config.py:129: The name tf.add_to_collection is deprecated. Please use tf.compat.v1.add_to_collection instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/config.py:131: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
# 
# Random Games Score Avg --> 92.86666666666666
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/layers/core.py:81: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/layers/core.py:145: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/initializations.py:174: calling TruncatedNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
# Instructions for updating:
# Call initializer instance with the dtype argument instead of passing it to the constructor
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/layers/core.py:239: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
# Instructions for updating:
# Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/optimizers.py:238: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/objectives.py:66: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
# Instructions for updating:
# keep_dims is deprecated, use keepdims instead
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/objectives.py:70: The name tf.log is deprecated. Please use tf.math.log instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/layers/estimator.py:189: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:571: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:115: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
# 
# 2020-05-19 23:55:41.140618: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
# 2020-05-19 23:55:41.157168: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.157402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
# name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
# pciBusID: 0000:01:00.0
# 2020-05-19 23:55:41.157561: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
# 2020-05-19 23:55:41.158488: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
# 2020-05-19 23:55:41.159344: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
# 2020-05-19 23:55:41.159541: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
# 2020-05-19 23:55:41.160644: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
# 2020-05-19 23:55:41.161475: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
# 2020-05-19 23:55:41.164125: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2020-05-19 23:55:41.164229: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.164476: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.164659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
# 2020-05-19 23:55:41.164905: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2020-05-19 23:55:41.191464: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2496000000 Hz
# 2020-05-19 23:55:41.191722: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4364090 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2020-05-19 23:55:41.191743: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2020-05-19 23:55:41.237272: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.237579: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4c05c20 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2020-05-19 23:55:41.237593: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1050, Compute Capability 6.1
# 2020-05-19 23:55:41.237746: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.237969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
# name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
# pciBusID: 0000:01:00.0
# 2020-05-19 23:55:41.238000: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
# 2020-05-19 23:55:41.238010: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
# 2020-05-19 23:55:41.238020: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
# 2020-05-19 23:55:41.238029: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
# 2020-05-19 23:55:41.238038: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
# 2020-05-19 23:55:41.238046: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
# 2020-05-19 23:55:41.238056: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2020-05-19 23:55:41.238089: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.238337: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.238524: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
# 2020-05-19 23:55:41.238546: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
# 2020-05-19 23:55:41.239163: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2020-05-19 23:55:41.239175: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
# 2020-05-19 23:55:41.239197: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
# 2020-05-19 23:55:41.239285: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.239511: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:41.239731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1382 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/summaries.py:46: The name tf.summary.scalar is deprecated. Please use tf.compat.v1.summary.scalar instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.where in 2.0, which has the same broadcast rule as np.where
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:134: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:164: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:165: The name tf.local_variables_initializer is deprecated. Please use tf.compat.v1.local_variables_initializer instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:166: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
# 
# WARNING:tensorflow:From /home/mijazz/pyProject/tfenv/lib/python3.6/site-packages/tflearn/helpers/trainer.py:167: The name tf.get_collection_ref is deprecated. Please use tf.compat.v1.get_collection_ref instead.
# 
# 2020-05-19 23:55:42.061238: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:42.061440: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
# name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
# pciBusID: 0000:01:00.0
# 2020-05-19 23:55:42.061471: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
# 2020-05-19 23:55:42.061481: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
# 2020-05-19 23:55:42.061488: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
# 2020-05-19 23:55:42.061496: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
# 2020-05-19 23:55:42.061503: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
# 2020-05-19 23:55:42.061511: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
# 2020-05-19 23:55:42.061519: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2020-05-19 23:55:42.061558: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:42.061733: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:42.061875: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
# 2020-05-19 23:55:42.061893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2020-05-19 23:55:42.061898: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
# 2020-05-19 23:55:42.061902: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
# 2020-05-19 23:55:42.061953: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:42.062122: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-05-19 23:55:42.062271: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1382 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
# ---------------------------------
# Run id: X76X9U
# Log directory: /tmp/tflearn_logs/
# ---------------------------------
# Training samples: 5512
# Validation samples: 0
# --
# 2020-05-19 23:55:42.352195: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
# Training Step: 1  | time: 0.342s
# | Adam | epoch: 001 | loss: 0.00000 - acc: 0.0000 -- iter: 0064/5512
# Training Step: 2  | total loss: 0.62383 | time: 0.346s
# | Adam | epoch: 001 | loss: 0.62383 - acc: 0.4500 -- iter: 0128/5512
# Training Step: 3  | total loss: 0.68043 | time: 0.350s
# | Adam | epoch: 001 | loss: 0.68043 - acc: 0.5293 -- iter: 0192/5512
# Training Step: 4  | total loss: 0.69029 | time: 0.354s
# | Adam | epoch: 001 | loss: 0.69029 - acc: 0.4604 -- iter: 0256/5512
# Training Step: 5  | total loss: 0.69190 | time: 0.357s
# | Adam | epoch: 001 | loss: 0.69190 - acc: 0.5419 -- iter: 0320/5512
# Training Step: 6  | total loss: 0.69327 | time: 0.361s
# | Adam | epoch: 001 | loss: 0.69327 - acc: 0.4547 -- iter: 0384/5512
# Training Step: 7  | total loss: 0.69280 | time: 0.364s
# | Adam | epoch: 001 | loss: 0.69280 - acc: 0.5288 -- iter: 0448/5512
# Training Step: 8  | total loss: 0.69308 | time: 0.367s
# | Adam | epoch: 001 | loss: 0.69308 - acc: 0.5038 -- iter: 0512/5512
# Training Step: 9  | total loss: 0.69279 | time: 0.371s
# | Adam | epoch: 001 | loss: 0.69279 - acc: 0.5349 -- iter: 0576/5512
# Training Step: 10  | total loss: 0.69296 | time: 0.374s
# | Adam | epoch: 001 | loss: 0.69296 - acc: 0.5174 -- iter: 0640/5512
# Training Step: 11  | total loss: 0.69387 | time: 0.380s
# | Adam | epoch: 001 | loss: 0.69387 - acc: 0.4426 -- iter: 0704/5512
# Training Step: 12  | total loss: 0.69364 | time: 0.383s
# | Adam | epoch: 001 | loss: 0.69364 - acc: 0.4614 -- iter: 0768/5512
# Training Step: 13  | total loss: 0.69358 | time: 0.387s
# | Adam | epoch: 001 | loss: 0.69358 - acc: 0.4645 -- iter: 0832/5512
# Training Step: 14  | total loss: 0.69334 | time: 0.390s
# | Adam | epoch: 001 | loss: 0.69334 - acc: 0.4854 -- iter: 0896/5512
# Training Step: 15  | total loss: 0.69322 | time: 0.393s
# | Adam | epoch: 001 | loss: 0.69322 - acc: 0.4972 -- iter: 0960/5512
# Training Step: 16  | total loss: 0.69314 | time: 0.396s
# | Adam | epoch: 001 | loss: 0.69314 - acc: 0.5041 -- iter: 1024/5512
# Training Step: 17  | total loss: 0.69350 | time: 0.399s
# | Adam | epoch: 001 | loss: 0.69350 - acc: 0.4577 -- iter: 1088/5512
# Training Step: 18  | total loss: 0.69325 | time: 0.403s
# | Adam | epoch: 001 | loss: 0.69325 - acc: 0.4885 -- iter: 1152/5512
# Training Step: 19  | total loss: 0.69324 | time: 0.407s
# | Adam | epoch: 001 | loss: 0.69324 - acc: 0.4871 -- iter: 1216/5512
# Training Step: 20  | total loss: 0.69303 | time: 0.410s
# | Adam | epoch: 001 | loss: 0.69303 - acc: 0.5214 -- iter: 1280/5512
# Training Step: 21  | total loss: 0.69300 | time: 0.413s
# | Adam | epoch: 001 | loss: 0.69300 - acc: 0.5245 -- iter: 1344/5512
# Training Step: 22  | total loss: 0.69298 | time: 0.416s
# | Adam | epoch: 001 | loss: 0.69298 - acc: 0.5218 -- iter: 1408/5512
# Training Step: 23  | total loss: 0.69308 | time: 0.419s
# | Adam | epoch: 001 | loss: 0.69308 - acc: 0.5064 -- iter: 1472/5512
# Training Step: 24  | total loss: 0.69283 | time: 0.422s
# | Adam | epoch: 001 | loss: 0.69283 - acc: 0.5398 -- iter: 1536/5512
# Training Step: 25  | total loss: 0.69290 | time: 0.426s
# | Adam | epoch: 001 | loss: 0.69290 - acc: 0.5247 -- iter: 1600/5512
# Training Step: 26  | total loss: 0.69295 | time: 0.429s
# | Adam | epoch: 001 | loss: 0.69295 - acc: 0.5181 -- iter: 1664/5512
# Training Step: 27  | total loss: 0.69302 | time: 0.432s
# | Adam | epoch: 001 | loss: 0.69302 - acc: 0.5054 -- iter: 1728/5512
# Training Step: 28  | total loss: 0.69294 | time: 0.437s
# | Adam | epoch: 001 | loss: 0.69294 - acc: 0.5041 -- iter: 1792/5512
# Training Step: 29  | total loss: 0.69298 | time: 0.441s
# | Adam | epoch: 001 | loss: 0.69298 - acc: 0.5031 -- iter: 1856/5512
# Training Step: 30  | total loss: 0.69294 | time: 0.444s
# | Adam | epoch: 001 | loss: 0.69294 - acc: 0.5061 -- iter: 1920/5512
# Training Step: 31  | total loss: 0.69323 | time: 0.449s
# | Adam | epoch: 001 | loss: 0.69323 - acc: 0.4758 -- iter: 1984/5512
# Training Step: 32  | total loss: 0.69283 | time: 0.452s
# | Adam | epoch: 001 | loss: 0.69283 - acc: 0.4953 -- iter: 2048/5512
# Training Step: 33  | total loss: 0.69247 | time: 0.455s
# | Adam | epoch: 001 | loss: 0.69247 - acc: 0.4963 -- iter: 2112/5512
# Training Step: 34  | total loss: 0.69247 | time: 0.459s
# | Adam | epoch: 001 | loss: 0.69247 - acc: 0.4971 -- iter: 2176/5512
# Training Step: 35  | total loss: 0.69196 | time: 0.462s
# | Adam | epoch: 001 | loss: 0.69196 - acc: 0.5075 -- iter: 2240/5512
# Training Step: 36  | total loss: 0.69247 | time: 0.464s
# | Adam | epoch: 001 | loss: 0.69247 - acc: 0.4772 -- iter: 2304/5512
# Training Step: 37  | total loss: 0.69179 | time: 0.469s
# | Adam | epoch: 001 | loss: 0.69179 - acc: 0.4818 -- iter: 2368/5512
# Training Step: 38  | total loss: 0.69102 | time: 0.473s
# | Adam | epoch: 001 | loss: 0.69102 - acc: 0.4915 -- iter: 2432/5512
# Training Step: 39  | total loss: 0.69181 | time: 0.476s
# | Adam | epoch: 001 | loss: 0.69181 - acc: 0.5140 -- iter: 2496/5512
# Training Step: 40  | total loss: 0.69156 | time: 0.480s
# | Adam | epoch: 001 | loss: 0.69156 - acc: 0.5378 -- iter: 2560/5512
# Training Step: 41  | total loss: 0.69033 | time: 0.483s
# | Adam | epoch: 001 | loss: 0.69033 - acc: 0.5394 -- iter: 2624/5512
# Training Step: 42  | total loss: 0.68852 | time: 0.486s
# | Adam | epoch: 001 | loss: 0.68852 - acc: 0.5548 -- iter: 2688/5512
# Training Step: 43  | total loss: 0.68735 | time: 0.489s
# | Adam | epoch: 001 | loss: 0.68735 - acc: 0.5700 -- iter: 2752/5512
# Training Step: 44  | total loss: 0.68464 | time: 0.492s
# | Adam | epoch: 001 | loss: 0.68464 - acc: 0.5714 -- iter: 2816/5512
# Training Step: 45  | total loss: 0.67907 | time: 0.495s
# | Adam | epoch: 001 | loss: 0.67907 - acc: 0.5938 -- iter: 2880/5512
# Training Step: 46  | total loss: 0.67357 | time: 0.498s
# | Adam | epoch: 001 | loss: 0.67357 - acc: 0.6042 -- iter: 2944/5512
# Training Step: 47  | total loss: 0.67491 | time: 0.502s
# | Adam | epoch: 001 | loss: 0.67491 - acc: 0.6127 -- iter: 3008/5512
# Training Step: 48  | total loss: 0.67451 | time: 0.505s
# | Adam | epoch: 001 | loss: 0.67451 - acc: 0.6247 -- iter: 3072/5512
# Training Step: 49  | total loss: 0.67153 | time: 0.508s
# | Adam | epoch: 001 | loss: 0.67153 - acc: 0.6174 -- iter: 3136/5512
# Training Step: 50  | total loss: 0.67600 | time: 0.511s
# | Adam | epoch: 001 | loss: 0.67600 - acc: 0.6088 -- iter: 3200/5512
# Training Step: 51  | total loss: 0.69265 | time: 0.514s
# | Adam | epoch: 001 | loss: 0.69265 - acc: 0.5875 -- iter: 3264/5512
# Training Step: 52  | total loss: 0.69029 | time: 0.517s
# | Adam | epoch: 001 | loss: 0.69029 - acc: 0.5908 -- iter: 3328/5512
# Training Step: 53  | total loss: 0.68904 | time: 0.520s
# | Adam | epoch: 001 | loss: 0.68904 - acc: 0.5889 -- iter: 3392/5512
# Training Step: 54  | total loss: 0.68893 | time: 0.523s
# | Adam | epoch: 001 | loss: 0.68893 - acc: 0.5896 -- iter: 3456/5512
# Training Step: 55  | total loss: 0.68266 | time: 0.526s
# | Adam | epoch: 001 | loss: 0.68266 - acc: 0.5947 -- iter: 3520/5512
# Training Step: 56  | total loss: 0.68302 | time: 0.529s
# | Adam | epoch: 001 | loss: 0.68302 - acc: 0.5857 -- iter: 3584/5512
# Training Step: 57  | total loss: 0.68263 | time: 0.532s
# | Adam | epoch: 001 | loss: 0.68263 - acc: 0.5869 -- iter: 3648/5512
# Training Step: 58  | total loss: 0.68658 | time: 0.535s
# | Adam | epoch: 001 | loss: 0.68658 - acc: 0.5750 -- iter: 3712/5512
# Training Step: 59  | total loss: 0.68456 | time: 0.538s
# | Adam | epoch: 001 | loss: 0.68456 - acc: 0.5838 -- iter: 3776/5512
# Training Step: 60  | total loss: 0.68291 | time: 0.541s
# | Adam | epoch: 001 | loss: 0.68291 - acc: 0.5831 -- iter: 3840/5512
# Training Step: 61  | total loss: 0.67919 | time: 0.544s
# | Adam | epoch: 001 | loss: 0.67919 - acc: 0.5885 -- iter: 3904/5512
# Training Step: 62  | total loss: 0.67767 | time: 0.547s
# | Adam | epoch: 001 | loss: 0.67767 - acc: 0.5993 -- iter: 3968/5512
# Training Step: 63  | total loss: 0.67552 | time: 0.550s
# | Adam | epoch: 001 | loss: 0.67552 - acc: 0.6065 -- iter: 4032/5512
# Training Step: 64  | total loss: 0.67408 | time: 0.554s
# | Adam | epoch: 001 | loss: 0.67408 - acc: 0.6127 -- iter: 4096/5512
# Training Step: 65  | total loss: 0.67220 | time: 0.557s
# | Adam | epoch: 001 | loss: 0.67220 - acc: 0.6181 -- iter: 4160/5512
# Training Step: 66  | total loss: 0.67419 | time: 0.560s
# | Adam | epoch: 001 | loss: 0.67419 - acc: 0.6075 -- iter: 4224/5512
# Training Step: 67  | total loss: 0.67126 | time: 0.563s
# | Adam | epoch: 001 | loss: 0.67126 - acc: 0.6171 -- iter: 4288/5512
# Training Step: 68  | total loss: 0.66874 | time: 0.566s
# | Adam | epoch: 001 | loss: 0.66874 - acc: 0.6199 -- iter: 4352/5512
# Training Step: 69  | total loss: 0.67047 | time: 0.569s
# | Adam | epoch: 001 | loss: 0.67047 - acc: 0.6095 -- iter: 4416/5512
# Training Step: 70  | total loss: 0.66813 | time: 0.572s
# | Adam | epoch: 001 | loss: 0.66813 - acc: 0.6131 -- iter: 4480/5512
# Training Step: 71  | total loss: 0.66518 | time: 0.576s
# | Adam | epoch: 001 | loss: 0.66518 - acc: 0.6198 -- iter: 4544/5512
# Training Step: 72  | total loss: 0.66602 | time: 0.582s
# | Adam | epoch: 001 | loss: 0.66602 - acc: 0.6222 -- iter: 4608/5512
# Training Step: 73  | total loss: 0.66725 | time: 0.585s
# | Adam | epoch: 001 | loss: 0.66725 - acc: 0.6173 -- iter: 4672/5512
# Training Step: 74  | total loss: 0.66713 | time: 0.589s
# | Adam | epoch: 001 | loss: 0.66713 - acc: 0.6130 -- iter: 4736/5512
# Training Step: 75  | total loss: 0.66588 | time: 0.592s
# | Adam | epoch: 001 | loss: 0.66588 - acc: 0.6126 -- iter: 4800/5512
# Training Step: 76  | total loss: 0.66480 | time: 0.595s
# | Adam | epoch: 001 | loss: 0.66480 - acc: 0.6122 -- iter: 4864/5512
# Training Step: 77  | total loss: 0.66901 | time: 0.601s
# | Adam | epoch: 001 | loss: 0.66901 - acc: 0.6053 -- iter: 4928/5512
# Training Step: 78  | total loss: 0.66854 | time: 0.605s
# | Adam | epoch: 001 | loss: 0.66854 - acc: 0.6106 -- iter: 4992/5512
# Training Step: 79  | total loss: 0.66081 | time: 0.608s
# | Adam | epoch: 001 | loss: 0.66081 - acc: 0.6186 -- iter: 5056/5512
# Training Step: 80  | total loss: 0.65743 | time: 0.613s
# | Adam | epoch: 001 | loss: 0.65743 - acc: 0.6193 -- iter: 5120/5512
# Training Step: 81  | total loss: 0.65728 | time: 0.617s
# | Adam | epoch: 001 | loss: 0.65728 - acc: 0.6198 -- iter: 5184/5512
# Training Step: 82  | total loss: 0.66007 | time: 0.620s
# | Adam | epoch: 001 | loss: 0.66007 - acc: 0.6203 -- iter: 5248/5512
# Training Step: 83  | total loss: 0.66876 | time: 0.624s
# | Adam | epoch: 001 | loss: 0.66876 - acc: 0.6083 -- iter: 5312/5512
# Training Step: 84  | total loss: 0.66684 | time: 0.628s
# | Adam | epoch: 001 | loss: 0.66684 - acc: 0.6115 -- iter: 5376/5512
# Training Step: 85  | total loss: 0.67116 | time: 0.632s
# | Adam | epoch: 001 | loss: 0.67116 - acc: 0.6020 -- iter: 5440/5512
# Training Step: 86  | total loss: 0.66731 | time: 0.639s
# | Adam | epoch: 001 | loss: 0.66731 - acc: 0.6058 -- iter: 5504/5512
# Training Step: 87  | total loss: 0.66644 | time: 0.643s
# | Adam | epoch: 001 | loss: 0.66644 - acc: 0.6062 -- iter: 5512/5512
# --
# Training Step: 88  | total loss: 0.65774 | time: 0.004s
# | Adam | epoch: 002 | loss: 0.65774 - acc: 0.6331 -- iter: 0064/5512
# Training Step: 89  | total loss: 0.64930 | time: 0.008s
# | Adam | epoch: 002 | loss: 0.64930 - acc: 0.6573 -- iter: 0128/5512
# Training Step: 90  | total loss: 0.65098 | time: 0.012s
# | Adam | epoch: 002 | loss: 0.65098 - acc: 0.6525 -- iter: 0192/5512
# Training Step: 91  | total loss: 0.65295 | time: 0.017s
# | Adam | epoch: 002 | loss: 0.65295 - acc: 0.6419 -- iter: 0256/5512
# Training Step: 92  | total loss: 0.65322 | time: 0.021s
# | Adam | epoch: 002 | loss: 0.65322 - acc: 0.6433 -- iter: 0320/5512
# Training Step: 93  | total loss: 0.65672 | time: 0.025s
# | Adam | epoch: 002 | loss: 0.65672 - acc: 0.6384 -- iter: 0384/5512
# Training Step: 94  | total loss: 0.66119 | time: 0.029s
# | Adam | epoch: 002 | loss: 0.66119 - acc: 0.6292 -- iter: 0448/5512
# Training Step: 95  | total loss: 0.65971 | time: 0.033s
# | Adam | epoch: 002 | loss: 0.65971 - acc: 0.6272 -- iter: 0512/5512
# Training Step: 96  | total loss: 0.66011 | time: 0.037s
# | Adam | epoch: 002 | loss: 0.66011 - acc: 0.6223 -- iter: 0576/5512
# Training Step: 97  | total loss: 0.66283 | time: 0.040s
# | Adam | epoch: 002 | loss: 0.66283 - acc: 0.6195 -- iter: 0640/5512
# Training Step: 98  | total loss: 0.65810 | time: 0.044s
# | Adam | epoch: 002 | loss: 0.65810 - acc: 0.6247 -- iter: 0704/5512
# Training Step: 99  | total loss: 0.65673 | time: 0.048s
# | Adam | epoch: 002 | loss: 0.65673 - acc: 0.6232 -- iter: 0768/5512
# Training Step: 100  | total loss: 0.65658 | time: 0.051s
# | Adam | epoch: 002 | loss: 0.65658 - acc: 0.6202 -- iter: 0832/5512
# Training Step: 101  | total loss: 0.65836 | time: 0.055s
# | Adam | epoch: 002 | loss: 0.65836 - acc: 0.6145 -- iter: 0896/5512
# Training Step: 102  | total loss: 0.65738 | time: 0.058s
# | Adam | epoch: 002 | loss: 0.65738 - acc: 0.6155 -- iter: 0960/5512
# Training Step: 103  | total loss: 0.65956 | time: 0.061s
# | Adam | epoch: 002 | loss: 0.65956 - acc: 0.6118 -- iter: 1024/5512
# Training Step: 104  | total loss: 0.65615 | time: 0.065s
# | Adam | epoch: 002 | loss: 0.65615 - acc: 0.6147 -- iter: 1088/5512
# Training Step: 105  | total loss: 0.65825 | time: 0.068s
# | Adam | epoch: 002 | loss: 0.65825 - acc: 0.6079 -- iter: 1152/5512
# Training Step: 106  | total loss: 0.65994 | time: 0.071s
# | Adam | epoch: 002 | loss: 0.65994 - acc: 0.6033 -- iter: 1216/5512
# Training Step: 107  | total loss: 0.66207 | time: 0.075s
# | Adam | epoch: 002 | loss: 0.66207 - acc: 0.6024 -- iter: 1280/5512
# Training Step: 108  | total loss: 0.65742 | time: 0.078s
# | Adam | epoch: 002 | loss: 0.65742 - acc: 0.6109 -- iter: 1344/5512
# Training Step: 109  | total loss: 0.66163 | time: 0.082s
# | Adam | epoch: 002 | loss: 0.66163 - acc: 0.6061 -- iter: 1408/5512
# Training Step: 110  | total loss: 0.65752 | time: 0.086s
# | Adam | epoch: 002 | loss: 0.65752 - acc: 0.6126 -- iter: 1472/5512
# Training Step: 111  | total loss: 0.65943 | time: 0.089s
# | Adam | epoch: 002 | loss: 0.65943 - acc: 0.6108 -- iter: 1536/5512
# Training Step: 112  | total loss: 0.66241 | time: 0.093s
# | Adam | epoch: 002 | loss: 0.66241 - acc: 0.6059 -- iter: 1600/5512
# Training Step: 113  | total loss: 0.66116 | time: 0.096s
# | Adam | epoch: 002 | loss: 0.66116 - acc: 0.6078 -- iter: 1664/5512
# Training Step: 114  | total loss: 0.65442 | time: 0.100s
# | Adam | epoch: 002 | loss: 0.65442 - acc: 0.6252 -- iter: 1728/5512
# Training Step: 115  | total loss: 0.65137 | time: 0.103s
# | Adam | epoch: 002 | loss: 0.65137 - acc: 0.6298 -- iter: 1792/5512
# Training Step: 116  | total loss: 0.65378 | time: 0.107s
# | Adam | epoch: 002 | loss: 0.65378 - acc: 0.6231 -- iter: 1856/5512
# Training Step: 117  | total loss: 0.65822 | time: 0.110s
# | Adam | epoch: 002 | loss: 0.65822 - acc: 0.6170 -- iter: 1920/5512
# Training Step: 118  | total loss: 0.66014 | time: 0.114s
# | Adam | epoch: 002 | loss: 0.66014 - acc: 0.6132 -- iter: 1984/5512
# Training Step: 119  | total loss: 0.66137 | time: 0.118s
# | Adam | epoch: 002 | loss: 0.66137 - acc: 0.6097 -- iter: 2048/5512
# Training Step: 120  | total loss: 0.66418 | time: 0.122s
# | Adam | epoch: 002 | loss: 0.66418 - acc: 0.6065 -- iter: 2112/5512
# Training Step: 121  | total loss: 0.66201 | time: 0.125s
# | Adam | epoch: 002 | loss: 0.66201 - acc: 0.6099 -- iter: 2176/5512
# Training Step: 122  | total loss: 0.65976 | time: 0.129s
# | Adam | epoch: 002 | loss: 0.65976 - acc: 0.6145 -- iter: 2240/5512
# Training Step: 123  | total loss: 0.65763 | time: 0.132s
# | Adam | epoch: 002 | loss: 0.65763 - acc: 0.6172 -- iter: 2304/5512
# Training Step: 124  | total loss: 0.65518 | time: 0.138s
# | Adam | epoch: 002 | loss: 0.65518 - acc: 0.6164 -- iter: 2368/5512
# Training Step: 125  | total loss: 0.65447 | time: 0.142s
# | Adam | epoch: 002 | loss: 0.65447 - acc: 0.6188 -- iter: 2432/5512
# Training Step: 126  | total loss: 0.65729 | time: 0.147s
# | Adam | epoch: 002 | loss: 0.65729 - acc: 0.6147 -- iter: 2496/5512
# Training Step: 127  | total loss: 0.65745 | time: 0.151s
# | Adam | epoch: 002 | loss: 0.65745 - acc: 0.6142 -- iter: 2560/5512
# Training Step: 128  | total loss: 0.65558 | time: 0.155s
# | Adam | epoch: 002 | loss: 0.65558 - acc: 0.6184 -- iter: 2624/5512
# Training Step: 129  | total loss: 0.65235 | time: 0.159s
# | Adam | epoch: 002 | loss: 0.65235 - acc: 0.6253 -- iter: 2688/5512
# Training Step: 130  | total loss: 0.65094 | time: 0.164s
# | Adam | epoch: 002 | loss: 0.65094 - acc: 0.6284 -- iter: 2752/5512
# Training Step: 131  | total loss: 0.64904 | time: 0.168s
# | Adam | epoch: 002 | loss: 0.64904 - acc: 0.6328 -- iter: 2816/5512
# Training Step: 132  | total loss: 0.64451 | time: 0.172s
# | Adam | epoch: 002 | loss: 0.64451 - acc: 0.6351 -- iter: 2880/5512
# Training Step: 133  | total loss: 0.64219 | time: 0.177s
# | Adam | epoch: 002 | loss: 0.64219 - acc: 0.6372 -- iter: 2944/5512
# Training Step: 134  | total loss: 0.63930 | time: 0.183s
# | Adam | epoch: 002 | loss: 0.63930 - acc: 0.6438 -- iter: 3008/5512
# Training Step: 135  | total loss: 0.64749 | time: 0.187s
# | Adam | epoch: 002 | loss: 0.64749 - acc: 0.6341 -- iter: 3072/5512
# Training Step: 136  | total loss: 0.65391 | time: 0.191s
# | Adam | epoch: 002 | loss: 0.65391 - acc: 0.6270 -- iter: 3136/5512
# Training Step: 137  | total loss: 0.65978 | time: 0.195s
# | Adam | epoch: 002 | loss: 0.65978 - acc: 0.6158 -- iter: 3200/5512
# Training Step: 138  | total loss: 0.65756 | time: 0.199s
# | Adam | epoch: 002 | loss: 0.65756 - acc: 0.6183 -- iter: 3264/5512
# Training Step: 139  | total loss: 0.65338 | time: 0.209s
# | Adam | epoch: 002 | loss: 0.65338 - acc: 0.6205 -- iter: 3328/5512
# Training Step: 140  | total loss: 0.65312 | time: 0.216s
# | Adam | epoch: 002 | loss: 0.65312 - acc: 0.6210 -- iter: 3392/5512
# Training Step: 141  | total loss: 0.66146 | time: 0.219s
# | Adam | epoch: 002 | loss: 0.66146 - acc: 0.6151 -- iter: 3456/5512
# Training Step: 142  | total loss: 0.65639 | time: 0.223s
# | Adam | epoch: 002 | loss: 0.65639 - acc: 0.6255 -- iter: 3520/5512
# Training Step: 143  | total loss: 0.66459 | time: 0.228s
# | Adam | epoch: 002 | loss: 0.66459 - acc: 0.6176 -- iter: 3584/5512
# Training Step: 144  | total loss: 0.66957 | time: 0.231s
# | Adam | epoch: 002 | loss: 0.66957 - acc: 0.6059 -- iter: 3648/5512
# Training Step: 145  | total loss: 0.66582 | time: 0.234s
# | Adam | epoch: 002 | loss: 0.66582 - acc: 0.6109 -- iter: 3712/5512
# Training Step: 146  | total loss: 0.66618 | time: 0.237s
# | Adam | epoch: 002 | loss: 0.66618 - acc: 0.6139 -- iter: 3776/5512
# Training Step: 147  | total loss: 0.66462 | time: 0.240s
# | Adam | epoch: 002 | loss: 0.66462 - acc: 0.6166 -- iter: 3840/5512
# Training Step: 148  | total loss: 0.66477 | time: 0.243s
# | Adam | epoch: 002 | loss: 0.66477 - acc: 0.6205 -- iter: 3904/5512
# Training Step: 149  | total loss: 0.66496 | time: 0.246s
# | Adam | epoch: 002 | loss: 0.66496 - acc: 0.6194 -- iter: 3968/5512
# Training Step: 150  | total loss: 0.66225 | time: 0.249s
# | Adam | epoch: 002 | loss: 0.66225 - acc: 0.6293 -- iter: 4032/5512
# Training Step: 151  | total loss: 0.66177 | time: 0.253s
# | Adam | epoch: 002 | loss: 0.66177 - acc: 0.6305 -- iter: 4096/5512
# Training Step: 152  | total loss: 0.66417 | time: 0.256s
# | Adam | epoch: 002 | loss: 0.66417 - acc: 0.6252 -- iter: 4160/5512
# Training Step: 153  | total loss: 0.66224 | time: 0.259s
# | Adam | epoch: 002 | loss: 0.66224 - acc: 0.6315 -- iter: 4224/5512
# Training Step: 154  | total loss: 0.66369 | time: 0.262s
# | Adam | epoch: 002 | loss: 0.66369 - acc: 0.6261 -- iter: 4288/5512
# Training Step: 155  | total loss: 0.66387 | time: 0.265s
# | Adam | epoch: 002 | loss: 0.66387 - acc: 0.6276 -- iter: 4352/5512
# Training Step: 156  | total loss: 0.66431 | time: 0.268s
# | Adam | epoch: 002 | loss: 0.66431 - acc: 0.6273 -- iter: 4416/5512
# Training Step: 157  | total loss: 0.66387 | time: 0.271s
# | Adam | epoch: 002 | loss: 0.66387 - acc: 0.6271 -- iter: 4480/5512
# Training Step: 158  | total loss: 0.66455 | time: 0.274s
# | Adam | epoch: 002 | loss: 0.66455 - acc: 0.6191 -- iter: 4544/5512
# Training Step: 159  | total loss: 0.66336 | time: 0.277s
# | Adam | epoch: 002 | loss: 0.66336 - acc: 0.6165 -- iter: 4608/5512
# Training Step: 160  | total loss: 0.66037 | time: 0.281s
# | Adam | epoch: 002 | loss: 0.66037 - acc: 0.6189 -- iter: 4672/5512
# Training Step: 161  | total loss: 0.66201 | time: 0.284s
# | Adam | epoch: 002 | loss: 0.66201 - acc: 0.6133 -- iter: 4736/5512
# Training Step: 162  | total loss: 0.65770 | time: 0.287s
# | Adam | epoch: 002 | loss: 0.65770 - acc: 0.6238 -- iter: 4800/5512
# Training Step: 163  | total loss: 0.65456 | time: 0.290s
# | Adam | epoch: 002 | loss: 0.65456 - acc: 0.6333 -- iter: 4864/5512
# Training Step: 164  | total loss: 0.65380 | time: 0.293s
# | Adam | epoch: 002 | loss: 0.65380 - acc: 0.6356 -- iter: 4928/5512
# Training Step: 165  | total loss: 0.65484 | time: 0.296s
# | Adam | epoch: 002 | loss: 0.65484 - acc: 0.6283 -- iter: 4992/5512
# Training Step: 166  | total loss: 0.66417 | time: 0.300s
# | Adam | epoch: 002 | loss: 0.66417 - acc: 0.6077 -- iter: 5056/5512
# Training Step: 167  | total loss: 0.66580 | time: 0.303s
# | Adam | epoch: 002 | loss: 0.66580 - acc: 0.6047 -- iter: 5120/5512
# Training Step: 168  | total loss: 0.66573 | time: 0.306s
# | Adam | epoch: 002 | loss: 0.66573 - acc: 0.6083 -- iter: 5184/5512
# Training Step: 169  | total loss: 0.67190 | time: 0.309s
# | Adam | epoch: 002 | loss: 0.67190 - acc: 0.6022 -- iter: 5248/5512
# Training Step: 170  | total loss: 0.66887 | time: 0.313s
# | Adam | epoch: 002 | loss: 0.66887 - acc: 0.6044 -- iter: 5312/5512
# Training Step: 171  | total loss: 0.66602 | time: 0.316s
# | Adam | epoch: 002 | loss: 0.66602 - acc: 0.6096 -- iter: 5376/5512
# Training Step: 172  | total loss: 0.66455 | time: 0.319s
# | Adam | epoch: 002 | loss: 0.66455 - acc: 0.6065 -- iter: 5440/5512
# Training Step: 173  | total loss: 0.67102 | time: 0.322s
# | Adam | epoch: 002 | loss: 0.67102 - acc: 0.5896 -- iter: 5504/5512
# Training Step: 174  | total loss: 0.66656 | time: 0.325s
# | Adam | epoch: 002 | loss: 0.66656 - acc: 0.5962 -- iter: 5512/5512
# --
# Training Step: 175  | total loss: 0.66160 | time: 0.003s
# | Adam | epoch: 003 | loss: 0.66160 - acc: 0.6085 -- iter: 0064/5512
# Training Step: 176  | total loss: 0.66123 | time: 0.006s
# | Adam | epoch: 003 | loss: 0.66123 - acc: 0.6101 -- iter: 0128/5512
# Training Step: 177  | total loss: 0.66064 | time: 0.018s
# | Adam | epoch: 003 | loss: 0.66064 - acc: 0.6116 -- iter: 0192/5512
# Training Step: 178  | total loss: 0.66388 | time: 0.021s
# | Adam | epoch: 003 | loss: 0.66388 - acc: 0.6067 -- iter: 0256/5512
# Training Step: 179  | total loss: 0.66443 | time: 0.025s
# | Adam | epoch: 003 | loss: 0.66443 - acc: 0.6070 -- iter: 0320/5512
# Training Step: 180  | total loss: 0.66470 | time: 0.028s
# | Adam | epoch: 003 | loss: 0.66470 - acc: 0.6072 -- iter: 0384/5512
# Training Step: 181  | total loss: 0.66732 | time: 0.032s
# | Adam | epoch: 003 | loss: 0.66732 - acc: 0.6090 -- iter: 0448/5512
# Training Step: 182  | total loss: 0.66791 | time: 0.036s
# | Adam | epoch: 003 | loss: 0.66791 - acc: 0.6059 -- iter: 0512/5512
# Training Step: 183  | total loss: 0.66137 | time: 0.040s
# | Adam | epoch: 003 | loss: 0.66137 - acc: 0.6172 -- iter: 0576/5512
# Training Step: 184  | total loss: 0.66046 | time: 0.045s
# | Adam | epoch: 003 | loss: 0.66046 - acc: 0.6149 -- iter: 0640/5512
# Training Step: 185  | total loss: 0.66271 | time: 0.049s
# | Adam | epoch: 003 | loss: 0.66271 - acc: 0.6049 -- iter: 0704/5512
# Training Step: 186  | total loss: 0.66192 | time: 0.052s
# | Adam | epoch: 003 | loss: 0.66192 - acc: 0.6085 -- iter: 0768/5512
# Training Step: 187  | total loss: 0.66057 | time: 0.056s
# | Adam | epoch: 003 | loss: 0.66057 - acc: 0.6101 -- iter: 0832/5512
# Training Step: 188  | total loss: 0.66366 | time: 0.061s
# | Adam | epoch: 003 | loss: 0.66366 - acc: 0.6069 -- iter: 0896/5512
# Training Step: 189  | total loss: 0.66181 | time: 0.064s
# | Adam | epoch: 003 | loss: 0.66181 - acc: 0.6072 -- iter: 0960/5512
# Training Step: 190  | total loss: 0.66540 | time: 0.068s
# | Adam | epoch: 003 | loss: 0.66540 - acc: 0.6012 -- iter: 1024/5512
# Training Step: 191  | total loss: 0.66454 | time: 0.072s
# | Adam | epoch: 003 | loss: 0.66454 - acc: 0.6020 -- iter: 1088/5512
# Training Step: 192  | total loss: 0.66346 | time: 0.075s
# | Adam | epoch: 003 | loss: 0.66346 - acc: 0.6074 -- iter: 1152/5512
# Training Step: 193  | total loss: 0.66903 | time: 0.078s
# | Adam | epoch: 003 | loss: 0.66903 - acc: 0.5967 -- iter: 1216/5512
# Training Step: 194  | total loss: 0.66809 | time: 0.082s
# | Adam | epoch: 003 | loss: 0.66809 - acc: 0.5979 -- iter: 1280/5512
# Training Step: 195  | total loss: 0.66449 | time: 0.084s
# | Adam | epoch: 003 | loss: 0.66449 - acc: 0.6069 -- iter: 1344/5512
# Training Step: 196  | total loss: 0.65923 | time: 0.088s
# | Adam | epoch: 003 | loss: 0.65923 - acc: 0.6150 -- iter: 1408/5512
# Training Step: 197  | total loss: 0.65424 | time: 0.093s
# | Adam | epoch: 003 | loss: 0.65424 - acc: 0.6269 -- iter: 1472/5512
# Training Step: 198  | total loss: 0.65005 | time: 0.096s
# | Adam | epoch: 003 | loss: 0.65005 - acc: 0.6330 -- iter: 1536/5512
# Training Step: 199  | total loss: 0.65092 | time: 0.099s
# | Adam | epoch: 003 | loss: 0.65092 - acc: 0.6306 -- iter: 1600/5512
# Training Step: 200  | total loss: 0.65842 | time: 0.103s
# | Adam | epoch: 003 | loss: 0.65842 - acc: 0.6144 -- iter: 1664/5512
# Training Step: 201  | total loss: 0.65459 | time: 0.106s
# | Adam | epoch: 003 | loss: 0.65459 - acc: 0.6248 -- iter: 1728/5512
# Training Step: 202  | total loss: 0.65688 | time: 0.109s
# | Adam | epoch: 003 | loss: 0.65688 - acc: 0.6233 -- iter: 1792/5512
# Training Step: 203  | total loss: 0.65666 | time: 0.112s
# | Adam | epoch: 003 | loss: 0.65666 - acc: 0.6188 -- iter: 1856/5512
# Training Step: 204  | total loss: 0.65522 | time: 0.115s
# | Adam | epoch: 003 | loss: 0.65522 - acc: 0.6225 -- iter: 1920/5512
# Training Step: 205  | total loss: 0.65500 | time: 0.119s
# | Adam | epoch: 003 | loss: 0.65500 - acc: 0.6243 -- iter: 1984/5512
# Training Step: 206  | total loss: 0.65583 | time: 0.122s
# | Adam | epoch: 003 | loss: 0.65583 - acc: 0.6260 -- iter: 2048/5512
# Training Step: 207  | total loss: 0.65266 | time: 0.125s
# | Adam | epoch: 003 | loss: 0.65266 - acc: 0.6306 -- iter: 2112/5512
# Training Step: 208  | total loss: 0.65132 | time: 0.128s
# | Adam | epoch: 003 | loss: 0.65132 - acc: 0.6331 -- iter: 2176/5512
# Training Step: 209  | total loss: 0.65578 | time: 0.131s
# | Adam | epoch: 003 | loss: 0.65578 - acc: 0.6276 -- iter: 2240/5512
# Training Step: 210  | total loss: 0.65818 | time: 0.134s
# | Adam | epoch: 003 | loss: 0.65818 - acc: 0.6242 -- iter: 2304/5512
# Training Step: 211  | total loss: 0.65596 | time: 0.137s
# | Adam | epoch: 003 | loss: 0.65596 - acc: 0.6243 -- iter: 2368/5512
# Training Step: 212  | total loss: 0.65815 | time: 0.140s
# | Adam | epoch: 003 | loss: 0.65815 - acc: 0.6244 -- iter: 2432/5512
# Training Step: 213  | total loss: 0.65288 | time: 0.143s
# | Adam | epoch: 003 | loss: 0.65288 - acc: 0.6354 -- iter: 2496/5512
# Training Step: 214  | total loss: 0.65417 | time: 0.146s
# | Adam | epoch: 003 | loss: 0.65417 - acc: 0.6359 -- iter: 2560/5512
# Training Step: 215  | total loss: 0.65707 | time: 0.149s
# | Adam | epoch: 003 | loss: 0.65707 - acc: 0.6223 -- iter: 2624/5512
# Training Step: 216  | total loss: 0.65455 | time: 0.152s
# | Adam | epoch: 003 | loss: 0.65455 - acc: 0.6273 -- iter: 2688/5512
# Training Step: 217  | total loss: 0.65567 | time: 0.156s
# | Adam | epoch: 003 | loss: 0.65567 - acc: 0.6192 -- iter: 2752/5512
# Training Step: 218  | total loss: 0.65937 | time: 0.159s
# | Adam | epoch: 003 | loss: 0.65937 - acc: 0.6198 -- iter: 2816/5512
# Training Step: 219  | total loss: 0.65628 | time: 0.162s
# | Adam | epoch: 003 | loss: 0.65628 - acc: 0.6235 -- iter: 2880/5512
# Training Step: 220  | total loss: 0.65623 | time: 0.165s
# | Adam | epoch: 003 | loss: 0.65623 - acc: 0.6220 -- iter: 2944/5512
# Training Step: 221  | total loss: 0.65559 | time: 0.168s
# | Adam | epoch: 003 | loss: 0.65559 - acc: 0.6239 -- iter: 3008/5512
# Training Step: 222  | total loss: 0.65662 | time: 0.171s
# | Adam | epoch: 003 | loss: 0.65662 - acc: 0.6209 -- iter: 3072/5512
# Training Step: 223  | total loss: 0.65730 | time: 0.174s
# | Adam | epoch: 003 | loss: 0.65730 - acc: 0.6244 -- iter: 3136/5512
# Training Step: 224  | total loss: 0.65687 | time: 0.177s
# | Adam | epoch: 003 | loss: 0.65687 - acc: 0.6245 -- iter: 3200/5512
# Training Step: 225  | total loss: 0.65065 | time: 0.180s
# | Adam | epoch: 003 | loss: 0.65065 - acc: 0.6323 -- iter: 3264/5512
# Training Step: 226  | total loss: 0.64831 | time: 0.183s
# | Adam | epoch: 003 | loss: 0.64831 - acc: 0.6363 -- iter: 3328/5512
# Training Step: 227  | total loss: 0.64937 | time: 0.186s
# | Adam | epoch: 003 | loss: 0.64937 - acc: 0.6336 -- iter: 3392/5512
# Training Step: 228  | total loss: 0.65087 | time: 0.190s
# | Adam | epoch: 003 | loss: 0.65087 - acc: 0.6374 -- iter: 3456/5512
# Training Step: 229  | total loss: 0.65806 | time: 0.192s
# | Adam | epoch: 003 | loss: 0.65806 - acc: 0.6268 -- iter: 3520/5512
# Training Step: 230  | total loss: 0.65790 | time: 0.196s
# | Adam | epoch: 003 | loss: 0.65790 - acc: 0.6219 -- iter: 3584/5512
# Training Step: 231  | total loss: 0.66239 | time: 0.199s
# | Adam | epoch: 003 | loss: 0.66239 - acc: 0.6160 -- iter: 3648/5512
# Training Step: 232  | total loss: 0.65883 | time: 0.202s
# | Adam | epoch: 003 | loss: 0.65883 - acc: 0.6185 -- iter: 3712/5512
# Training Step: 233  | total loss: 0.66032 | time: 0.205s
# | Adam | epoch: 003 | loss: 0.66032 - acc: 0.6144 -- iter: 3776/5512
# Training Step: 234  | total loss: 0.65700 | time: 0.208s
# | Adam | epoch: 003 | loss: 0.65700 - acc: 0.6170 -- iter: 3840/5512
# Training Step: 235  | total loss: 0.66263 | time: 0.213s
# | Adam | epoch: 003 | loss: 0.66263 - acc: 0.6085 -- iter: 3904/5512
# Training Step: 236  | total loss: 0.66216 | time: 0.216s
# | Adam | epoch: 003 | loss: 0.66216 - acc: 0.5992 -- iter: 3968/5512
# Training Step: 237  | total loss: 0.66305 | time: 0.220s
# | Adam | epoch: 003 | loss: 0.66305 - acc: 0.6018 -- iter: 4032/5512
# Training Step: 238  | total loss: 0.65979 | time: 0.223s
# | Adam | epoch: 003 | loss: 0.65979 - acc: 0.6103 -- iter: 4096/5512
# Training Step: 239  | total loss: 0.66027 | time: 0.226s
# | Adam | epoch: 003 | loss: 0.66027 - acc: 0.6118 -- iter: 4160/5512
# Training Step: 240  | total loss: 0.66169 | time: 0.229s
# | Adam | epoch: 003 | loss: 0.66169 - acc: 0.6147 -- iter: 4224/5512
# Training Step: 241  | total loss: 0.66274 | time: 0.232s
# | Adam | epoch: 003 | loss: 0.66274 - acc: 0.6110 -- iter: 4288/5512
# Training Step: 242  | total loss: 0.66667 | time: 0.237s
# | Adam | epoch: 003 | loss: 0.66667 - acc: 0.6046 -- iter: 4352/5512
# Training Step: 243  | total loss: 0.66677 | time: 0.241s
# | Adam | epoch: 003 | loss: 0.66677 - acc: 0.6051 -- iter: 4416/5512
# Training Step: 244  | total loss: 0.67027 | time: 0.244s
# | Adam | epoch: 003 | loss: 0.67027 - acc: 0.5993 -- iter: 4480/5512
# Training Step: 245  | total loss: 0.66947 | time: 0.249s
# | Adam | epoch: 003 | loss: 0.66947 - acc: 0.6003 -- iter: 4544/5512
# Training Step: 246  | total loss: 0.67052 | time: 0.253s
# | Adam | epoch: 003 | loss: 0.67052 - acc: 0.5996 -- iter: 4608/5512
# Training Step: 247  | total loss: 0.66805 | time: 0.256s
# | Adam | epoch: 003 | loss: 0.66805 - acc: 0.6084 -- iter: 4672/5512
# Training Step: 248  | total loss: 0.66883 | time: 0.260s
# | Adam | epoch: 003 | loss: 0.66883 - acc: 0.6023 -- iter: 4736/5512
# Training Step: 249  | total loss: 0.66761 | time: 0.263s
# | Adam | epoch: 003 | loss: 0.66761 - acc: 0.5998 -- iter: 4800/5512
# Training Step: 250  | total loss: 0.66610 | time: 0.268s
# | Adam | epoch: 003 | loss: 0.66610 - acc: 0.6008 -- iter: 4864/5512
# Training Step: 251  | total loss: 0.66507 | time: 0.272s
# | Adam | epoch: 003 | loss: 0.66507 - acc: 0.6048 -- iter: 4928/5512
# Training Step: 252  | total loss: 0.66348 | time: 0.275s
# | Adam | epoch: 003 | loss: 0.66348 - acc: 0.6084 -- iter: 4992/5512
# Training Step: 253  | total loss: 0.66309 | time: 0.279s
# | Adam | epoch: 003 | loss: 0.66309 - acc: 0.6100 -- iter: 5056/5512
# Training Step: 254  | total loss: 0.66143 | time: 0.283s
# | Adam | epoch: 003 | loss: 0.66143 - acc: 0.6115 -- iter: 5120/5512
# Training Step: 255  | total loss: 0.66756 | time: 0.286s
# | Adam | epoch: 003 | loss: 0.66756 - acc: 0.6004 -- iter: 5184/5512
# Training Step: 256  | total loss: 0.66564 | time: 0.289s
# | Adam | epoch: 003 | loss: 0.66564 - acc: 0.6028 -- iter: 5248/5512
# Training Step: 257  | total loss: 0.66356 | time: 0.294s
# | Adam | epoch: 003 | loss: 0.66356 - acc: 0.6051 -- iter: 5312/5512
# Training Step: 258  | total loss: 0.66398 | time: 0.297s
# | Adam | epoch: 003 | loss: 0.66398 - acc: 0.5992 -- iter: 5376/5512
# Training Step: 259  | total loss: 0.66448 | time: 0.300s
# | Adam | epoch: 003 | loss: 0.66448 - acc: 0.5987 -- iter: 5440/5512
# Training Step: 260  | total loss: 0.66232 | time: 0.303s
# | Adam | epoch: 003 | loss: 0.66232 - acc: 0.6060 -- iter: 5504/5512
# Training Step: 261  | total loss: 0.66279 | time: 0.306s
# | Adam | epoch: 003 | loss: 0.66279 - acc: 0.6017 -- iter: 5512/5512
# --
# Training Step: 262  | total loss: 0.66127 | time: 0.003s
# | Adam | epoch: 004 | loss: 0.66127 - acc: 0.6024 -- iter: 0064/5512
# Training Step: 263  | total loss: 0.66192 | time: 0.006s
# | Adam | epoch: 004 | loss: 0.66192 - acc: 0.6016 -- iter: 0128/5512
# Training Step: 264  | total loss: 0.66821 | time: 0.009s
# | Adam | epoch: 004 | loss: 0.66821 - acc: 0.5914 -- iter: 0192/5512
# Training Step: 265  | total loss: 0.67123 | time: 0.012s
# | Adam | epoch: 004 | loss: 0.67123 - acc: 0.5823 -- iter: 0256/5512
# Training Step: 266  | total loss: 0.66673 | time: 0.015s
# | Adam | epoch: 004 | loss: 0.66673 - acc: 0.5881 -- iter: 0320/5512
# Training Step: 267  | total loss: 0.66689 | time: 0.018s
# | Adam | epoch: 004 | loss: 0.66689 - acc: 0.5855 -- iter: 0384/5512
# Training Step: 268  | total loss: 0.66341 | time: 0.021s
# | Adam | epoch: 004 | loss: 0.66341 - acc: 0.5910 -- iter: 0448/5512
# Training Step: 269  | total loss: 0.66140 | time: 0.024s
# | Adam | epoch: 004 | loss: 0.66140 - acc: 0.5976 -- iter: 0512/5512
# Training Step: 270  | total loss: 0.67128 | time: 0.027s
# | Adam | epoch: 004 | loss: 0.67128 - acc: 0.5956 -- iter: 0576/5512
# Training Step: 271  | total loss: 0.66997 | time: 0.030s
# | Adam | epoch: 004 | loss: 0.66997 - acc: 0.5986 -- iter: 0640/5512
# Training Step: 272  | total loss: 0.66559 | time: 0.033s
# | Adam | epoch: 004 | loss: 0.66559 - acc: 0.6059 -- iter: 0704/5512
# Training Step: 273  | total loss: 0.66651 | time: 0.036s
# | Adam | epoch: 004 | loss: 0.66651 - acc: 0.6016 -- iter: 0768/5512
# Training Step: 274  | total loss: 0.66804 | time: 0.039s
# | Adam | epoch: 004 | loss: 0.66804 - acc: 0.5976 -- iter: 0832/5512
# Training Step: 275  | total loss: 0.66146 | time: 0.043s
# | Adam | epoch: 004 | loss: 0.66146 - acc: 0.6066 -- iter: 0896/5512
# Training Step: 276  | total loss: 0.66590 | time: 0.046s
# | Adam | epoch: 004 | loss: 0.66590 - acc: 0.6038 -- iter: 0960/5512
# Training Step: 277  | total loss: 0.67008 | time: 0.049s
# | Adam | epoch: 004 | loss: 0.67008 - acc: 0.6012 -- iter: 1024/5512
# Training Step: 278  | total loss: 0.66609 | time: 0.052s
# | Adam | epoch: 004 | loss: 0.66609 - acc: 0.6052 -- iter: 1088/5512
# Training Step: 279  | total loss: 0.66807 | time: 0.054s
# | Adam | epoch: 004 | loss: 0.66807 - acc: 0.5978 -- iter: 1152/5512
# Training Step: 280  | total loss: 0.66567 | time: 0.058s
# | Adam | epoch: 004 | loss: 0.66567 - acc: 0.6005 -- iter: 1216/5512
# Training Step: 281  | total loss: 0.66873 | time: 0.061s
# | Adam | epoch: 004 | loss: 0.66873 - acc: 0.5998 -- iter: 1280/5512
# Training Step: 282  | total loss: 0.66699 | time: 0.064s
# | Adam | epoch: 004 | loss: 0.66699 - acc: 0.5961 -- iter: 1344/5512
# Training Step: 283  | total loss: 0.66378 | time: 0.067s
# | Adam | epoch: 004 | loss: 0.66378 - acc: 0.6162 -- iter: 1408/5512
# Training Step: 284  | total loss: 0.66563 | time: 0.070s
# | Adam | epoch: 004 | loss: 0.66563 - acc: 0.6108 -- iter: 1472/5512
# Training Step: 285  | total loss: 0.66542 | time: 0.073s
# | Adam | epoch: 004 | loss: 0.66542 - acc: 0.6122 -- iter: 1536/5512
# Training Step: 286  | total loss: 0.66480 | time: 0.076s
# | Adam | epoch: 004 | loss: 0.66480 - acc: 0.6213 -- iter: 1600/5512
# Training Step: 287  | total loss: 0.66151 | time: 0.079s
# | Adam | epoch: 004 | loss: 0.66151 - acc: 0.6342 -- iter: 1664/5512
# Training Step: 288  | total loss: 0.66275 | time: 0.083s
# | Adam | epoch: 004 | loss: 0.66275 - acc: 0.6286 -- iter: 1728/5512
# Training Step: 289  | total loss: 0.66314 | time: 0.086s
# | Adam | epoch: 004 | loss: 0.66314 - acc: 0.6251 -- iter: 1792/5512
# Training Step: 290  | total loss: 0.66116 | time: 0.089s
# | Adam | epoch: 004 | loss: 0.66116 - acc: 0.6298 -- iter: 1856/5512
# Training Step: 291  | total loss: 0.66297 | time: 0.092s
# | Adam | epoch: 004 | loss: 0.66297 - acc: 0.6246 -- iter: 1920/5512
# Training Step: 292  | total loss: 0.66213 | time: 0.095s
# | Adam | epoch: 004 | loss: 0.66213 - acc: 0.6278 -- iter: 1984/5512
# Training Step: 293  | total loss: 0.66228 | time: 0.098s
# | Adam | epoch: 004 | loss: 0.66228 - acc: 0.6228 -- iter: 2048/5512
# Training Step: 294  | total loss: 0.65460 | time: 0.101s
# | Adam | epoch: 004 | loss: 0.65460 - acc: 0.6449 -- iter: 2112/5512
# Training Step: 295  | total loss: 0.64837 | time: 0.104s
# | Adam | epoch: 004 | loss: 0.64837 - acc: 0.6538 -- iter: 2176/5512
# Training Step: 296  | total loss: 0.64471 | time: 0.108s
# | Adam | epoch: 004 | loss: 0.64471 - acc: 0.6619 -- iter: 2240/5512
# Training Step: 297  | total loss: 0.64643 | time: 0.112s
# | Adam | epoch: 004 | loss: 0.64643 - acc: 0.6598 -- iter: 2304/5512
# Training Step: 298  | total loss: 0.64630 | time: 0.115s
# | Adam | epoch: 004 | loss: 0.64630 - acc: 0.6579 -- iter: 2368/5512
# Training Step: 299  | total loss: 0.64923 | time: 0.119s
# | Adam | epoch: 004 | loss: 0.64923 - acc: 0.6483 -- iter: 2432/5512
# Training Step: 300  | total loss: 0.64949 | time: 0.122s
# | Adam | epoch: 004 | loss: 0.64949 - acc: 0.6476 -- iter: 2496/5512
# Training Step: 301  | total loss: 0.65055 | time: 0.125s
# | Adam | epoch: 004 | loss: 0.65055 - acc: 0.6453 -- iter: 2560/5512
# Training Step: 302  | total loss: 0.64955 | time: 0.128s
# | Adam | epoch: 004 | loss: 0.64955 - acc: 0.6480 -- iter: 2624/5512
# Training Step: 303  | total loss: 0.65100 | time: 0.131s
# | Adam | epoch: 004 | loss: 0.65100 - acc: 0.6410 -- iter: 2688/5512
# Training Step: 304  | total loss: 0.65370 | time: 0.137s
# | Adam | epoch: 004 | loss: 0.65370 - acc: 0.6347 -- iter: 2752/5512
# Training Step: 305  | total loss: 0.65202 | time: 0.140s
# | Adam | epoch: 004 | loss: 0.65202 - acc: 0.6306 -- iter: 2816/5512
# Training Step: 306  | total loss: 0.65256 | time: 0.143s
# | Adam | epoch: 004 | loss: 0.65256 - acc: 0.6332 -- iter: 2880/5512
# Training Step: 307  | total loss: 0.65510 | time: 0.147s
# | Adam | epoch: 004 | loss: 0.65510 - acc: 0.6292 -- iter: 2944/5512
# Training Step: 308  | total loss: 0.65341 | time: 0.150s
# | Adam | epoch: 004 | loss: 0.65341 - acc: 0.6335 -- iter: 3008/5512
# Training Step: 309  | total loss: 0.64919 | time: 0.153s
# | Adam | epoch: 004 | loss: 0.64919 - acc: 0.6420 -- iter: 3072/5512
# Training Step: 310  | total loss: 0.65358 | time: 0.158s
# | Adam | epoch: 004 | loss: 0.65358 - acc: 0.6387 -- iter: 3136/5512
# Training Step: 311  | total loss: 0.65057 | time: 0.161s
# | Adam | epoch: 004 | loss: 0.65057 - acc: 0.6452 -- iter: 3200/5512
# Training Step: 312  | total loss: 0.65456 | time: 0.164s
# | Adam | epoch: 004 | loss: 0.65456 - acc: 0.6432 -- iter: 3264/5512
# Training Step: 313  | total loss: 0.65393 | time: 0.169s
# | Adam | epoch: 004 | loss: 0.65393 - acc: 0.6460 -- iter: 3328/5512
# Training Step: 314  | total loss: 0.64882 | time: 0.172s
# | Adam | epoch: 004 | loss: 0.64882 - acc: 0.6502 -- iter: 3392/5512
# Training Step: 315  | total loss: 0.65174 | time: 0.175s
# | Adam | epoch: 004 | loss: 0.65174 - acc: 0.6492 -- iter: 3456/5512
# Training Step: 316  | total loss: 0.65500 | time: 0.179s
# | Adam | epoch: 004 | loss: 0.65500 - acc: 0.6437 -- iter: 3520/5512
# Training Step: 317  | total loss: 0.65204 | time: 0.182s
# | Adam | epoch: 004 | loss: 0.65204 - acc: 0.6449 -- iter: 3584/5512
# Training Step: 318  | total loss: 0.65791 | time: 0.185s
# | Adam | epoch: 004 | loss: 0.65791 - acc: 0.6304 -- iter: 3648/5512
# Training Step: 319  | total loss: 0.66114 | time: 0.191s
# | Adam | epoch: 004 | loss: 0.66114 - acc: 0.6236 -- iter: 3712/5512
# Training Step: 320  | total loss: 0.66111 | time: 0.195s
# | Adam | epoch: 004 | loss: 0.66111 - acc: 0.6222 -- iter: 3776/5512
# Training Step: 321  | total loss: 0.66010 | time: 0.198s
# | Adam | epoch: 004 | loss: 0.66010 - acc: 0.6225 -- iter: 3840/5512
# Training Step: 322  | total loss: 0.65658 | time: 0.202s
# | Adam | epoch: 004 | loss: 0.65658 - acc: 0.6243 -- iter: 3904/5512
# Training Step: 323  | total loss: 0.65641 | time: 0.205s
# | Adam | epoch: 004 | loss: 0.65641 - acc: 0.6244 -- iter: 3968/5512
# Training Step: 324  | total loss: 0.65427 | time: 0.208s
# | Adam | epoch: 004 | loss: 0.65427 - acc: 0.6323 -- iter: 4032/5512
# Training Step: 325  | total loss: 0.65553 | time: 0.211s
# | Adam | epoch: 004 | loss: 0.65553 - acc: 0.6284 -- iter: 4096/5512
# Training Step: 326  | total loss: 0.65860 | time: 0.214s
# | Adam | epoch: 004 | loss: 0.65860 - acc: 0.6234 -- iter: 4160/5512
# Training Step: 327  | total loss: 0.65795 | time: 0.217s
# | Adam | epoch: 004 | loss: 0.65795 - acc: 0.6220 -- iter: 4224/5512
# Training Step: 328  | total loss: 0.65895 | time: 0.221s
# | Adam | epoch: 004 | loss: 0.65895 - acc: 0.6223 -- iter: 4288/5512
# Training Step: 329  | total loss: 0.65604 | time: 0.224s
# | Adam | epoch: 004 | loss: 0.65604 - acc: 0.6288 -- iter: 4352/5512
# Training Step: 330  | total loss: 0.66231 | time: 0.227s
# | Adam | epoch: 004 | loss: 0.66231 - acc: 0.6097 -- iter: 4416/5512
# Training Step: 331  | total loss: 0.66406 | time: 0.230s
# | Adam | epoch: 004 | loss: 0.66406 - acc: 0.6081 -- iter: 4480/5512
# Training Step: 332  | total loss: 0.65884 | time: 0.233s
# | Adam | epoch: 004 | loss: 0.65884 - acc: 0.6145 -- iter: 4544/5512
# Training Step: 333  | total loss: 0.65997 | time: 0.236s
# | Adam | epoch: 004 | loss: 0.65997 - acc: 0.6077 -- iter: 4608/5512
# Training Step: 334  | total loss: 0.66127 | time: 0.239s
# | Adam | epoch: 004 | loss: 0.66127 - acc: 0.6047 -- iter: 4672/5512
# Training Step: 335  | total loss: 0.65372 | time: 0.242s
# | Adam | epoch: 004 | loss: 0.65372 - acc: 0.6177 -- iter: 4736/5512
# Training Step: 336  | total loss: 0.65475 | time: 0.245s
# | Adam | epoch: 004 | loss: 0.65475 - acc: 0.6184 -- iter: 4800/5512
# Training Step: 337  | total loss: 0.65335 | time: 0.248s
# | Adam | epoch: 004 | loss: 0.65335 - acc: 0.6222 -- iter: 4864/5512
# Training Step: 338  | total loss: 0.65572 | time: 0.252s
# | Adam | epoch: 004 | loss: 0.65572 - acc: 0.6194 -- iter: 4928/5512
# Training Step: 339  | total loss: 0.65195 | time: 0.255s
# | Adam | epoch: 004 | loss: 0.65195 - acc: 0.6262 -- iter: 4992/5512
# Training Step: 340  | total loss: 0.65642 | time: 0.257s
# | Adam | epoch: 004 | loss: 0.65642 - acc: 0.6183 -- iter: 5056/5512
# Training Step: 341  | total loss: 0.65517 | time: 0.261s
# | Adam | epoch: 004 | loss: 0.65517 - acc: 0.6174 -- iter: 5120/5512
# Training Step: 342  | total loss: 0.65516 | time: 0.264s
# | Adam | epoch: 004 | loss: 0.65516 - acc: 0.6213 -- iter: 5184/5512
# Training Step: 343  | total loss: 0.65658 | time: 0.267s
# | Adam | epoch: 004 | loss: 0.65658 - acc: 0.6169 -- iter: 5248/5512
# Training Step: 344  | total loss: 0.65516 | time: 0.270s
# | Adam | epoch: 004 | loss: 0.65516 - acc: 0.6146 -- iter: 5312/5512
# Training Step: 345  | total loss: 0.64986 | time: 0.274s
# | Adam | epoch: 004 | loss: 0.64986 - acc: 0.6235 -- iter: 5376/5512
# Training Step: 346  | total loss: 0.65448 | time: 0.277s
# | Adam | epoch: 004 | loss: 0.65448 - acc: 0.6158 -- iter: 5440/5512
# Training Step: 347  | total loss: 0.65118 | time: 0.280s
# | Adam | epoch: 004 | loss: 0.65118 - acc: 0.6199 -- iter: 5504/5512
# Training Step: 348  | total loss: 0.64946 | time: 0.284s
# | Adam | epoch: 004 | loss: 0.64946 - acc: 0.6172 -- iter: 5512/5512
# --
# Training Step: 349  | total loss: 0.64597 | time: 0.003s
# | Adam | epoch: 005 | loss: 0.64597 - acc: 0.6227 -- iter: 0064/5512
# Training Step: 350  | total loss: 0.64528 | time: 0.007s
# | Adam | epoch: 005 | loss: 0.64528 - acc: 0.6292 -- iter: 0128/5512
# Training Step: 351  | total loss: 0.64507 | time: 0.010s
# | Adam | epoch: 005 | loss: 0.64507 - acc: 0.6288 -- iter: 0192/5512
# Training Step: 352  | total loss: 0.63700 | time: 0.014s
# | Adam | epoch: 005 | loss: 0.63700 - acc: 0.6409 -- iter: 0256/5512
# Training Step: 353  | total loss: 0.62811 | time: 0.017s
# | Adam | epoch: 005 | loss: 0.62811 - acc: 0.6518 -- iter: 0320/5512
# Training Step: 354  | total loss: 0.62648 | time: 0.020s
# | Adam | epoch: 005 | loss: 0.62648 - acc: 0.6569 -- iter: 0384/5512
# Training Step: 355  | total loss: 0.62751 | time: 0.023s
# | Adam | epoch: 005 | loss: 0.62751 - acc: 0.6506 -- iter: 0448/5512
# Training Step: 356  | total loss: 0.62430 | time: 0.027s
# | Adam | epoch: 005 | loss: 0.62430 - acc: 0.6512 -- iter: 0512/5512
# Training Step: 357  | total loss: 0.62206 | time: 0.032s
# | Adam | epoch: 005 | loss: 0.62206 - acc: 0.6564 -- iter: 0576/5512
# Training Step: 358  | total loss: 0.62277 | time: 0.035s
# | Adam | epoch: 005 | loss: 0.62277 - acc: 0.6564 -- iter: 0640/5512
# Training Step: 359  | total loss: 0.63548 | time: 0.039s
# | Adam | epoch: 005 | loss: 0.63548 - acc: 0.6517 -- iter: 0704/5512
# Training Step: 360  | total loss: 0.64245 | time: 0.044s
# | Adam | epoch: 005 | loss: 0.64245 - acc: 0.6427 -- iter: 0768/5512
# Training Step: 361  | total loss: 0.65183 | time: 0.047s
# | Adam | epoch: 005 | loss: 0.65183 - acc: 0.6347 -- iter: 0832/5512
# Training Step: 362  | total loss: 0.65270 | time: 0.050s
# | Adam | epoch: 005 | loss: 0.65270 - acc: 0.6353 -- iter: 0896/5512
# Training Step: 363  | total loss: 0.65279 | time: 0.054s
# | Adam | epoch: 005 | loss: 0.65279 - acc: 0.6327 -- iter: 0960/5512
# Training Step: 364  | total loss: 0.65321 | time: 0.057s
# | Adam | epoch: 005 | loss: 0.65321 - acc: 0.6366 -- iter: 1024/5512
# Training Step: 365  | total loss: 0.66367 | time: 0.060s
# | Adam | epoch: 005 | loss: 0.66367 - acc: 0.6198 -- iter: 1088/5512
# Training Step: 366  | total loss: 0.66345 | time: 0.064s
# | Adam | epoch: 005 | loss: 0.66345 - acc: 0.6110 -- iter: 1152/5512
# Training Step: 367  | total loss: 0.66021 | time: 0.067s
# | Adam | epoch: 005 | loss: 0.66021 - acc: 0.6171 -- iter: 1216/5512
# Training Step: 368  | total loss: 0.66326 | time: 0.070s
# | Adam | epoch: 005 | loss: 0.66326 - acc: 0.6069 -- iter: 1280/5512
# Training Step: 369  | total loss: 0.66797 | time: 0.073s
# | Adam | epoch: 005 | loss: 0.66797 - acc: 0.5962 -- iter: 1344/5512
# Training Step: 370  | total loss: 0.66604 | time: 0.078s
# | Adam | epoch: 005 | loss: 0.66604 - acc: 0.6038 -- iter: 1408/5512
# Training Step: 371  | total loss: 0.66236 | time: 0.081s
# | Adam | epoch: 005 | loss: 0.66236 - acc: 0.6059 -- iter: 1472/5512
# Training Step: 372  | total loss: 0.66029 | time: 0.084s
# | Adam | epoch: 005 | loss: 0.66029 - acc: 0.6110 -- iter: 1536/5512
# Training Step: 373  | total loss: 0.66019 | time: 0.088s
# | Adam | epoch: 005 | loss: 0.66019 - acc: 0.6186 -- iter: 1600/5512
# Training Step: 374  | total loss: 0.66188 | time: 0.091s
# | Adam | epoch: 005 | loss: 0.66188 - acc: 0.6114 -- iter: 1664/5512
# Training Step: 375  | total loss: 0.65938 | time: 0.094s
# | Adam | epoch: 005 | loss: 0.65938 - acc: 0.6190 -- iter: 1728/5512
# Training Step: 376  | total loss: 0.66334 | time: 0.097s
# | Adam | epoch: 005 | loss: 0.66334 - acc: 0.6181 -- iter: 1792/5512
# Training Step: 377  | total loss: 0.66707 | time: 0.101s
# | Adam | epoch: 005 | loss: 0.66707 - acc: 0.6094 -- iter: 1856/5512
# Training Step: 378  | total loss: 0.66692 | time: 0.104s
# | Adam | epoch: 005 | loss: 0.66692 - acc: 0.6156 -- iter: 1920/5512
# Training Step: 379  | total loss: 0.66817 | time: 0.107s
# | Adam | epoch: 005 | loss: 0.66817 - acc: 0.6056 -- iter: 1984/5512
# Training Step: 380  | total loss: 0.66938 | time: 0.111s
# | Adam | epoch: 005 | loss: 0.66938 - acc: 0.5966 -- iter: 2048/5512
# Training Step: 381  | total loss: 0.67355 | time: 0.114s
# | Adam | epoch: 005 | loss: 0.67355 - acc: 0.5948 -- iter: 2112/5512
# Training Step: 382  | total loss: 0.66896 | time: 0.117s
# | Adam | epoch: 005 | loss: 0.66896 - acc: 0.6041 -- iter: 2176/5512
# Training Step: 383  | total loss: 0.67033 | time: 0.121s
# | Adam | epoch: 005 | loss: 0.67033 - acc: 0.6030 -- iter: 2240/5512
# Training Step: 384  | total loss: 0.66802 | time: 0.124s
# | Adam | epoch: 005 | loss: 0.66802 - acc: 0.6068 -- iter: 2304/5512
# Training Step: 385  | total loss: 0.67373 | time: 0.127s
# | Adam | epoch: 005 | loss: 0.67373 - acc: 0.5945 -- iter: 2368/5512
# Training Step: 386  | total loss: 0.67146 | time: 0.130s
# | Adam | epoch: 005 | loss: 0.67146 - acc: 0.6023 -- iter: 2432/5512
# Training Step: 387  | total loss: 0.66870 | time: 0.133s
# | Adam | epoch: 005 | loss: 0.66870 - acc: 0.6092 -- iter: 2496/5512
# Training Step: 388  | total loss: 0.67079 | time: 0.136s
# | Adam | epoch: 005 | loss: 0.67079 - acc: 0.6061 -- iter: 2560/5512
# Training Step: 389  | total loss: 0.67050 | time: 0.139s
# | Adam | epoch: 005 | loss: 0.67050 - acc: 0.6080 -- iter: 2624/5512
# Training Step: 390  | total loss: 0.66709 | time: 0.142s
# | Adam | epoch: 005 | loss: 0.66709 - acc: 0.6175 -- iter: 2688/5512
# Training Step: 391  | total loss: 0.67061 | time: 0.146s
# | Adam | epoch: 005 | loss: 0.67061 - acc: 0.6120 -- iter: 2752/5512
# Training Step: 392  | total loss: 0.67028 | time: 0.149s
# | Adam | epoch: 005 | loss: 0.67028 - acc: 0.6118 -- iter: 2816/5512
# Training Step: 393  | total loss: 0.66663 | time: 0.152s
# | Adam | epoch: 005 | loss: 0.66663 - acc: 0.6209 -- iter: 2880/5512
# Training Step: 394  | total loss: 0.66807 | time: 0.155s
# | Adam | epoch: 005 | loss: 0.66807 - acc: 0.6135 -- iter: 2944/5512
# Training Step: 395  | total loss: 0.66797 | time: 0.158s
# | Adam | epoch: 005 | loss: 0.66797 - acc: 0.6131 -- iter: 3008/5512
# Training Step: 396  | total loss: 0.67227 | time: 0.161s
# | Adam | epoch: 005 | loss: 0.67227 - acc: 0.6033 -- iter: 3072/5512
# Training Step: 397  | total loss: 0.67342 | time: 0.164s
# | Adam | epoch: 005 | loss: 0.67342 - acc: 0.6024 -- iter: 3136/5512
# Training Step: 398  | total loss: 0.66957 | time: 0.167s
# | Adam | epoch: 005 | loss: 0.66957 - acc: 0.6062 -- iter: 3200/5512
# Training Step: 399  | total loss: 0.66786 | time: 0.170s
# | Adam | epoch: 005 | loss: 0.66786 - acc: 0.6050 -- iter: 3264/5512
# Training Step: 400  | total loss: 0.66379 | time: 0.173s
# | Adam | epoch: 005 | loss: 0.66379 - acc: 0.6070 -- iter: 3328/5512
# Training Step: 401  | total loss: 0.66387 | time: 0.176s
# | Adam | epoch: 005 | loss: 0.66387 - acc: 0.6056 -- iter: 3392/5512
# Training Step: 402  | total loss: 0.66296 | time: 0.179s
# | Adam | epoch: 005 | loss: 0.66296 - acc: 0.6107 -- iter: 3456/5512
# Training Step: 403  | total loss: 0.66129 | time: 0.182s
# | Adam | epoch: 005 | loss: 0.66129 - acc: 0.6168 -- iter: 3520/5512
# Training Step: 404  | total loss: 0.65812 | time: 0.185s
# | Adam | epoch: 005 | loss: 0.65812 - acc: 0.6223 -- iter: 3584/5512
# Training Step: 405  | total loss: 0.66319 | time: 0.189s
# | Adam | epoch: 005 | loss: 0.66319 - acc: 0.6163 -- iter: 3648/5512
# Training Step: 406  | total loss: 0.66483 | time: 0.192s
# | Adam | epoch: 005 | loss: 0.66483 - acc: 0.6172 -- iter: 3712/5512
# Training Step: 407  | total loss: 0.66722 | time: 0.195s
# | Adam | epoch: 005 | loss: 0.66722 - acc: 0.6102 -- iter: 3776/5512
# Training Step: 408  | total loss: 0.66185 | time: 0.198s
# | Adam | epoch: 005 | loss: 0.66185 - acc: 0.6210 -- iter: 3840/5512
# Training Step: 409  | total loss: 0.65358 | time: 0.201s
# | Adam | epoch: 005 | loss: 0.65358 - acc: 0.6339 -- iter: 3904/5512
# Training Step: 410  | total loss: 0.65650 | time: 0.204s
# | Adam | epoch: 005 | loss: 0.65650 - acc: 0.6315 -- iter: 3968/5512
# Training Step: 411  | total loss: 0.65247 | time: 0.207s
# | Adam | epoch: 005 | loss: 0.65247 - acc: 0.6371 -- iter: 4032/5512
# Training Step: 412  | total loss: 0.65126 | time: 0.210s
# | Adam | epoch: 005 | loss: 0.65126 - acc: 0.6374 -- iter: 4096/5512
# Training Step: 413  | total loss: 0.64842 | time: 0.213s
# | Adam | epoch: 005 | loss: 0.64842 - acc: 0.6378 -- iter: 4160/5512
# Training Step: 414  | total loss: 0.64902 | time: 0.217s
# | Adam | epoch: 005 | loss: 0.64902 - acc: 0.6365 -- iter: 4224/5512
# Training Step: 415  | total loss: 0.64961 | time: 0.220s
# | Adam | epoch: 005 | loss: 0.64961 - acc: 0.6322 -- iter: 4288/5512
# Training Step: 416  | total loss: 0.65951 | time: 0.223s
# | Adam | epoch: 005 | loss: 0.65951 - acc: 0.6205 -- iter: 4352/5512
# Training Step: 417  | total loss: 0.65415 | time: 0.226s
# | Adam | epoch: 005 | loss: 0.65415 - acc: 0.6304 -- iter: 4416/5512
# Training Step: 418  | total loss: 0.65699 | time: 0.230s
# | Adam | epoch: 005 | loss: 0.65699 - acc: 0.6267 -- iter: 4480/5512
# Training Step: 419  | total loss: 0.66122 | time: 0.233s
# | Adam | epoch: 005 | loss: 0.66122 - acc: 0.6203 -- iter: 4544/5512
# Training Step: 420  | total loss: 0.66241 | time: 0.238s
# | Adam | epoch: 005 | loss: 0.66241 - acc: 0.6223 -- iter: 4608/5512
# Training Step: 421  | total loss: 0.65723 | time: 0.242s
# | Adam | epoch: 005 | loss: 0.65723 - acc: 0.6288 -- iter: 4672/5512
# Training Step: 422  | total loss: 0.65953 | time: 0.245s
# | Adam | epoch: 005 | loss: 0.65953 - acc: 0.6285 -- iter: 4736/5512
# Training Step: 423  | total loss: 0.65940 | time: 0.251s
# | Adam | epoch: 005 | loss: 0.65940 - acc: 0.6281 -- iter: 4800/5512
# Training Step: 424  | total loss: 0.65555 | time: 0.254s
# | Adam | epoch: 005 | loss: 0.65555 - acc: 0.6294 -- iter: 4864/5512
# Training Step: 425  | total loss: 0.65651 | time: 0.258s
# | Adam | epoch: 005 | loss: 0.65651 - acc: 0.6274 -- iter: 4928/5512
# Training Step: 426  | total loss: 0.65770 | time: 0.262s
# | Adam | epoch: 005 | loss: 0.65770 - acc: 0.6209 -- iter: 4992/5512
# Training Step: 427  | total loss: 0.65514 | time: 0.267s
# | Adam | epoch: 005 | loss: 0.65514 - acc: 0.6275 -- iter: 5056/5512
# Training Step: 428  | total loss: 0.65402 | time: 0.271s
# | Adam | epoch: 005 | loss: 0.65402 - acc: 0.6242 -- iter: 5120/5512
# Training Step: 429  | total loss: 0.65111 | time: 0.274s
# | Adam | epoch: 005 | loss: 0.65111 - acc: 0.6242 -- iter: 5184/5512
# Training Step: 430  | total loss: 0.65092 | time: 0.277s
# | Adam | epoch: 005 | loss: 0.65092 - acc: 0.6228 -- iter: 5248/5512
# Training Step: 431  | total loss: 0.65469 | time: 0.282s
# | Adam | epoch: 005 | loss: 0.65469 - acc: 0.6214 -- iter: 5312/5512
# Training Step: 432  | total loss: 0.65545 | time: 0.286s
# | Adam | epoch: 005 | loss: 0.65545 - acc: 0.6280 -- iter: 5376/5512
# Training Step: 433  | total loss: 0.65564 | time: 0.290s
# | Adam | epoch: 005 | loss: 0.65564 - acc: 0.6277 -- iter: 5440/5512
# Training Step: 434  | total loss: 0.65631 | time: 0.293s
# | Adam | epoch: 005 | loss: 0.65631 - acc: 0.6196 -- iter: 5504/5512
# Training Step: 435  | total loss: 0.65775 | time: 0.297s
# | Adam | epoch: 005 | loss: 0.65775 - acc: 0.6170 -- iter: 5512/5512
# --
# Predicted Games Score Avg --> 199.0
# 
# Process finished with exit code 0
# 
# 
# ```
