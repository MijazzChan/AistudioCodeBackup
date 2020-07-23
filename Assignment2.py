#!/usr/bin/env python
# coding: utf-8

# In[4]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[5]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# 1 在Kaggle上下载数据集https://www.kaggle.com/xvivancos/barcelona-data-sets#accidents_2017.csv,并利用学到的python知识对该数据集进行特征分析，特征包括但不限定为下面内容：
# 1）对数据进行归一化处理
# 2）统计星期一到星期日的案件发生数，并画出直方图
# 3）猜测下受伤最严重的案件在时间或者空间分布上有什么特征

# In[6]:


import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt


# #### 项目依赖引用

# In[7]:


data = pd.read_csv('./work/accidents_2017.csv')
data.head(5)


# In[8]:


data_extract_weekday = data['Weekday'].value_counts()
dataFrame_weekday = pd.DataFrame(data_extract_weekday)
print(dataFrame_weekday.sort_values(by = 'Weekday'))


# #### 抽离出Weekday列的数据，并排列显示得上述结果，以下进行归一化处理。

# In[9]:


dataFrame_weekday_percent = dataFrame_weekday / sum(dataFrame_weekday['Weekday'])
print(dataFrame_weekday_percent.sort_values(by = 'Weekday'))


# #### 归一处理后，画直方图与占比图

# In[10]:


plt.barh(range(7), dataFrame_weekday['Weekday'], height=0.7, alpha=0.8)
plt.yticks(range(7), dataFrame_weekday.index)
plt.xlim(800, 1800)
plt.xlabel('Accidents Count')
plt.title('Distribution of Accidents')
plt.show()


# In[11]:


plt.pie(dataFrame_weekday_percent['Weekday'], labels=dataFrame_weekday_percent.index, startangle=90, shadow=True)
plt.axis('equal')
plt.legend()
plt.show()


# In[ ]:




