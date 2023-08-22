#!/user/bin python
#coding=UTF-8
'''
@author  : Eikken
#@file   : 隐马尔科夫.py
#@time   : 2019-05-23 12:49:51
'''

import numpy as np
import hmmlearn.hmm as hmm
import math

status = ['吃', '睡']  # 状态序列
observation = ['哭', '没精神', '找妈妈']  # 观测序列
n_status = len(status)
n_observation = len(observation)
start_probability = np.array([0.3, 0.7])  # 初始状态分布
# 状态转移概率矩阵
transition_probability = np.array([
    [0.1, 0.9],
    [0.8, 0.2]
])
# 观测生成矩阵
emission__probability = np.array([
    [0.7, 0.1, 0.2],
    [0.3, 0.5, 0.2]
])
# HMM模型构建
model = hmm.MultinomialHMM(n_components=n_status)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission__probability
# 行为模型
Actions = np.array([[0, 1, 2]])
Action_model = Actions.T

score = model.score(Action_model, lengths=None)
Action = ','.join(map(lambda x: observation[x], Actions[0]))
print("\t\"", Action, "\"的概率为：", end='')
print('\t',math.exp(score) * 100, '%')

# 所有观测值状态转移概率
predict_proba = model.predict_proba(Action_model, lengths=None)
# 维特比算法估计最可能的状态
logprob, behavior = model.decode(Action_model, algorithm="viterbi")
print("\t观察状态:", ','.join(map(lambda x: observation[x], Actions[0])))
print("\t观测值状态转移的概率为：", status, "：")
for i in Actions[0]:
    print("\t"*2, observation[i], ':', predict_proba[i])
print("\t状态如下:", ','.join(map(lambda x: status[x], behavior)))
print("\t对应概率：", math.exp(logprob) * 100, '%')
