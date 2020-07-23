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
