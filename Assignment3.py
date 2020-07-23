#!/usr/bin/env python
# coding: utf-8

# In[48]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[49]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


#  第二章PPT和视频中讲解了A*算法原理，以及跨越障碍物的例子，请用Python实现它，要求：<br/>
#  1.地图的大小可任意，障碍物的数量至少是2，长度和位置可任意固定。
#  2.在估算H值的时候，分别采用欧式距离和曼哈顿距离两种方式，并比较两种方法给出的最优路径是否相同
#  3.可不画出图，文字打印路径作为结果。

# ## A star Algorithm
# **2017326603075 陈浩骏 17信科1班**
# 参考资料(可能需要proxy)：
# + [StackOverFlow](https://stackoverflow.com/questions/1332466/how-does-dijkstras-algorithm-and-a-star-compare) A* search algorithm
# + [WikiPedia](https://en.wikipedia.org/wiki/A*_search_algorithm) Introduction to A* From Amit’s Thoughts on Pathfinding
# + [Stanford](http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html) how-does-dijkstras-algorithm-and-a-star-compare

# In[50]:


import math


# 依赖导入

# In[51]:


class Node():
    def __init__(self, parentNode=None, position=None):
        self.parentNode = parentNode
        self.position = position
        self.f = 0
        self.g = 0
        self.h = 0
    
    def __eq__(self, other):
        return self.position == other.position


# ### 面向对象：Node-节点类
# - `parentNode` 为某一点的双亲结点，具有指向性，便于回溯，可避免使用栈或队列记录路线
# - `position` 为当前节点在图中的位置，`type=tuple`，二元位置`(pos_x, pos_y)`
# - `f,g,h` 分别为A* 算法中的对应距离代价cost，详情看A* 算法
# - 重载`__eq__`函数，比较两点相同仅需看位置
# 
# 因需使用两种距离算法`Manhattan` `Eclidean`，为了可读性，使用函数调用算法会比脚本序更好。

# In[52]:


def aStar(maze, startNodePosition, endNodePosition, diagonally=True, euclidean=True):
    '''
        maze: 2D-list(0/1) 1 is block, 0 is route.
        diagonally: 允许是否能对角行走
        TODO: Mijazz_Chan on Mar 11th, Diagonally 为 False 会遇到死循环，修正
        euclidean: Flag of Euclidean distance, if False, use Manhattan instead.
        Configuration of A*
    '''
    openList = []
    closedList = []

    allowCrossDiagonally = diagonally
    isEuclideanDistance = euclidean
    # 8个方向 东南西北， 根据是否能对角，取slice
    direction = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighborNode = direction if allowCrossDiagonally else direction[:4]
    # 距离算法
    euclideanDistance = lambda pos1_x, pos2_x, pos1_y, pos2_y: math.sqrt((pos1_x - pos2_x) ** 2 + (pos1_y - pos2_y) ** 2)
    manhattanDistance = lambda pos1_x, pos2_x, pos1_y, pos2_y: abs(pos1_x - pos2_x) + abs(pos1_y - pos2_y)
    calDistance = euclideanDistance if isEuclideanDistance else manhattanDistance

    # Mark:AVOID TO USE NON-DIAGONALLY, ENDLESS ITERATION! - TODO:Mijazz_Chan on Mar 10th
    startNode = Node(None, startNodePosition)
    startNode.f = startNode.g = startNode.h = 0
    endNode = Node(None, endNodePosition)
    endNode.f = endNode.g = endNode.h = 0

    openList.append(startNode)

    while(len(openList) > 0):
        currentNode = openList[0]
        ci = 0
        # 在openList取最小f的点
        for i, cnode in enumerate(openList):
            if cnode.f < currentNode.f:
                currentNode = cnode
                ci = i
        # 从openList中取出，并放入closedList
        openList.pop(ci)
        closedList.append(currentNode)

        # 假设当前currentNode为endNode，回溯每个点并返回路径
        if currentNode == endNode:
            print('Cost-> %.3f' % float(currentNode.f))
            path = []
            current = currentNode
            while current is not None:
                path.append(current.position)
                current = current.parentNode
            return path[::-1]

        children = []
        # newPostion为临近方向，加（1,0）即为东边的邻节点，加(-1, -1)即为西北的邻节点
        for newPosition in neighborNode:
            nodePosition = (currentNode.position[0] + newPosition[0], currentNode.position[1] + newPosition[1])
            # 界内和障碍物判定
            if nodePosition[0] > (len(maze) - 1) or nodePosition[0] < 0 or nodePosition[1] > (len(maze[len(maze) - 1]) - 1) or nodePosition[1] < 0:
                continue
            if maze[nodePosition[0]][nodePosition[1]] != 0:
                continue 
            # 该节点的非障碍物，非界外的孩子节点
            newNode = Node(currentNode, nodePosition)
            children.append(newNode)
        
        for child in children:
            # 如果孩子节点已经被遍历过，丢弃
            if child in closedList:
                continue
            # 对角移动，g加1.4, 横向纵向移动g加1
            if abs(child.position[0] - currentNode.position[0]) + abs(child.position[1] - currentNode.position[1]) == 2:
                child.g = currentNode.g + 1.4
            else:
                child.g = currentNode.g + 1
            child.h = calDistance(child.position[0], endNode.position[0], child.position[1], endNode.position[1])
            child.f = child.g + child.h

            # 查看是否需要更新g
            for openNode in openList:
                if child == openNode and child.g > openNode.g:
                    continue
            openList.append(child)


# ### 算法调用
# + `maze` 为矩阵形式的二维数组，该格式用例为10x10的矩阵，0为可通行Block，1为障碍物Block，存在四个障碍物，终点位于第四障碍物后
# + `startNode` 起点， `endNode`终点
# + `diagonally` 允许对角移动
# + `Euclidean` 欧氏距离或曼哈顿距离Boolean Flag
# **使用10x10的测试用例(10x10, 4障碍)，在不允许对角移动时会有死循环出现** <br>.
# 
# **使用6x6的测试用例(6x6, 3障碍)，均正常**

# In[53]:


# 10x10测试用例 
maze10 = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

start = (0, 0)
end = (5, 9)

path = aStar(maze10, start, end, diagonally=True, euclidean=True)
print('Euclidean True, Diagonally True', path)

path = aStar(maze10, start, end, diagonally=True, euclidean=False)
print('Manhattan True, Diagonally True', path)


# In[54]:


maze6 = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]]
end = (4, 4)
path = aStar(maze6, start, end, diagonally=False, euclidean=True)
print('Euclidean True, Diagonally False', path)
path = aStar(maze6, start, end, diagonally=True, euclidean=True)
print('Euclidean True, Diagonally True', path)
path = aStar(maze6, start, end, diagonally=False, euclidean=False)
print('Euclidean False, Diagonally False', path)
path = aStar(maze6, start, end, diagonally=True, euclidean=False)
print('Euclidean False, Diagonally True', path)


# ### 实验结论
# + 在10x10测试用例与6x6测试用例中，均实现欧氏距离与曼哈顿距离的比较，其路径在`6x6,不允许对角移动`出现了不同，但cost都是一样的。
# + 在实验时，`maze`增大时，相应的空间需要和时间需要会激增，并且在解决路径问题上，开放性的道路，即切合实际的导航行驶，启发式搜索会降低很多复杂度，但是对于障碍物多时，即迷宫问题是，我的理解是深度或广度遍历算法会反而较快。
