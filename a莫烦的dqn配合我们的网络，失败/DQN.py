
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

# import pickle
import copy


# 本地类
import mlp



class DQN(object):
    def __init__(self,mlp_architecture):


        # mlp_architecture 是一个列表，描述了要求的神经网络的结构 每层几个神经元
        # N_STATES 是输入层神经元的个数
        # N_ACTIONS 是输出层神经元的个数
        
        self.MEMORY_CAPACITY = 2**12
        self.N_STATES = mlp_architecture[0]
        self.N_ACTIONS = mlp_architecture[-1]
        lr = 0.01

        mlp_architecture = [4,32,2]
        self.eval_net, self.target_net = mlp.mlp(mlp_architecture,lr), mlp.mlp(mlp_architecture,lr)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        
        
        
        
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr)
        self.loss_func = nn.MSELoss()

        #pid
        self.integral = 0  # 位移的积分
        self.max_i = 2**6 # 位移积分的上界
        self.ki= -3 #-1

        self.it = 0
        # self.max_it = 2**6
        self.kit = -3

    def random_action(self):

        N_ACTIONS = self.N_ACTIONS
        action = np.random.randint(0, N_ACTIONS)
        return action
    

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 1).cuda().half()

        # print(x.shape)

        EPSILON = self.epsilon
        ENV_A_SHAPE = 0


        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)


            xxx = torch.max(actions_value, 1)[1]

            xxx = xxx.cpu()
            action = xxx.data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = self.random_action()
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)


        return action

    def store_transition(self, state, a, r, s_):

        MEMORY_CAPACITY=self.MEMORY_CAPACITY

        transition = np.hstack((state, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print('dqn.learn()')
        TARGET_REPLACE_ITER = 2**7
        MEMORY_CAPACITY = self.MEMORY_CAPACITY
        BATCH_SIZE = 3 #2**3
        N_STATES = self.N_STATES


        GAMMA = 0.9


        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # self.target_net.load_state_dict(self.eval_net.state_dict())
            self.target_net= copy.deepcopy(self.eval_net)

        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])


        # q_eval w.r.t the action in experience
        b_a = b_a.cuda()

        
        b_r = b_r.permute(1,0)     
        b_r = b_r.cuda().half()
        
        
        b_s = b_s.permute(1,0)     
        b_s = b_s.cuda().half()



        b_a =  b_a.permute(1,0)  

        b_s_ =  b_s_.permute(1,0).cuda().half()

        # print('shape')
        # print(b_s.shape)
        # print(b_a.shape)
        # print(b_s_.shape)
        # print(b_r.shape)






        # print(b_s.shape)
        # print(b_s)
        xxx = self.eval_net.forward(b_s)

        # print(b_a.shape)
        q_eval = xxx.gather(0, b_a)  # shape (batch, 1)

        # print('xxx',xxx)
        # print('ac',b_a)
        # print('qv',q_eval)

        # print(b_s_.shape)
        q_next = self.target_net.test(b_s_)
        
        yyy = torch.max(q_next, 0)[0].reshape((1,BATCH_SIZE))
        
        # .detach()     # detach from graph, don't backpropagate
        
        # yyy=q_next.max(1)[0].view(BATCH_SIZE, 1)


        q_target = b_r + GAMMA * yyy   # shape (batch, 1)

        
        # q_target =  q_target.permute(1,0)  
        # print('qn',q_next)
        # print('yyy',yyy)
        # print('b_r',b_r)
        # print('qt',q_target)



        q_target = q_target.half()

        loss = self.loss_func(q_eval, q_target)

        # self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.step()

        self.eval_net.update()

        fl = float(loss)
        return loss


    def get_i(self):
        # 解码
        integral = self.integral      
        error = self.error

        # 计算积分
        # integral += error
        error = abs(error)

        alpha = 0.5
        # integral = (1-alpha)*integral + alpha*error
        integral = (1-alpha)*integral + error

        # 无界变换成有界有多种函数 ，这里随便写一种简单的。不是我们的重点。

        # print(integral)
        if(abs(integral)>self.max_i):
            print('积分爆了')
            if integral>0:
                integral = self.max_i
            else:
                integral = -self.max_i

        # 更新
        self.integral = integral
        return integral

    def reward_f(self,next_state):

        x, x_dot, theta, theta_dot = next_state

        # pid算法
        # self.error = x
        self.error = abs(x)**2
        integral = self.get_i()
        ri = integral* self.ki
        
        self.it = 0.7*self.it + abs(theta)**2
        rit = self.it *self.kit
        # rit = 0
        
        r0 = 0.7
        r1 = -abs(x)/2.4
        r2 = -abs(theta)/0.209

        reward = r0 +r1 + r2 + ri + rit
        if(reward< 2**-10):reward = 2**-10

        reward = reward**2

        return reward
    