#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 有50件物品，其价格和重量在文件thing.txt中。
# 假设某个货车最大载重量为1000千克,请使用遗传算法求解火车所能装下的最大价值物品清单。


# ### 第八次作业
# 2017326603075   陈浩骏   17信科1班
# 
# ___
# 
# 从文件中取数据， 参数初始化用`numpy`的`arrange(1, 51)`生成50长度的一维数组作为染色体。

# In[2]:


import random as rd
from random import randint
import numpy as np

weight = []
value = []

with open('./work/thing.csv', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        value.append(int(line.split(',')[0]))
        weight.append(int(line.split(',')[1]))

# 染色体
item_number = np.arange(1, 51)
weight = np.array(weight)
value = np.array(value)
weight_threshold = 1000
print('Item  Weight  Value')
for i in range(item_number.shape[0]):
    print('{0}     {1}     {2}'.format(item_number[i], weight[i], value[i]))


# #### 染色体初始化
# 数量为8，每一个染色体代表一个解
# #### 迭代次数=60000
# 
# 注：`np.random.randint(2, size=pop_size)`才可以取到[1, 0, 1, ...]

# In[3]:


solutions_per_pop = 8
pop_size = (solutions_per_pop, item_number.shape[0])
initial_population = np.random.randint(2, size=pop_size)
initial_population = initial_population.astype(int)
num_generations = 60000
print('Initial population: \n{}'.format(initial_population))


# In[4]:


# 适应度计算
def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else:
            fitness[i] = 0
    return fitness.astype(int)

# 选择或淘汰
def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i, :] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents

# 交叉（嫁接）
def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1] / 2)
    crossover_rate = 0.9
    i = 0
    while parents.shape[0] < num_offsprings:
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parents.shape[0]
        offsprings[i, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
        offsprings[i, crossover_point:] = parents[parent2_index, crossover_point:]
        i = +1
    return offsprings

# 基因位变异
def mutation(offsprings):
    mutants = np.empty(offsprings.shape)
    # 变异率
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i, :] = offsprings[i, :]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0, offsprings.shape[1] - 1)
        if mutants[i, int_random_value] == 0:
            mutants[i, int_random_value] = 1
        else:
            mutants[i, int_random_value] = 0
    return mutants


# 流式函数
def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0] / 2)
    num_offsprings = pop_size[0] - num_parents
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print('Last generation: \n{}\n'.format(population))
    fitness_last_gen = cal_fitness(weight, value, population, threshold)
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0], :])
    return parameters, fitness_history


parameters, fitness_history = optimize(weight, value, initial_population, pop_size, num_generations, weight_threshold)
print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
selected_items = item_number * parameters
print('result as follow')
for i in range(selected_items.shape[1]):
    if selected_items[0][i] != 0:
        print('{}'.format(selected_items[0][i]))

