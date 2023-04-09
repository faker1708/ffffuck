import sys
import pygame
import time
import random
import dqn
import torch
import numpy as np

import matplotlib.pyplot as plt
import pickle

# 多人竞技
# 非零和，与环境 变化现在还没有。
# 现在是最简单的版本，一只羊，一头狼，没有技能 ，速度相同 。感觉羊稳赢。


class role():

    def __init__(role_a,color='blue'):
        rs = pygame.Surface((2**4,2**4),flags=pygame.HWSURFACE)
        rs.fill(color)
        role_a.x = random.randint(0,2**8.-1) #774 #
        role_a.y =  random.randint(0,2**8-1) #774 #
        role_a.surface = rs
        role_a.flag = 0
        
        # role_a.screen = screen

    def move(role_a):
        # 
        # role_a.x +=1


        view = role_a.view

        view = torch.tensor(view)
        view = view.reshape((1,4))
        # print('____________vs',view.shape)
        # print('v',view)
        # view = torch.unsqueeze(view, 1)
        # view = torch.unsqueeze(view, 1)
        # view = view.long()
        # view = torch.tensor(view,dtype=torch.float32)
        view = view.type(torch.float32)
        

        # print(view.dtype)
        # 
        # print('viewlllllllllllll',view)
        action = role_a.dqn.choose_action(view)
        # print(action)

        if(role_a.flag == 0):
            # print(action)
            action  = 4

        
        role_a.action = action  # 保存下

        world_x = role_a.game.wold_x
        world_y = role_a.game.wold_y

        # print(action)

        # 循环位移 ，遇到边界就传送到另一边
        # 有限，无边界

        # print('ac',action)
        if(action ==0):# up
            role_a.x -=1
            
            if(role_a.x<=-1):
                role_a.x = world_x


        elif(action ==1):#down
            role_a.x+=1
            # print('up')
            # print('role',role_a.x,world_x)
            
            if(role_a.x>=world_x):
                role_a.x = 0


        elif(action ==2):# left
            role_a.y-=1
            
            if(role_a.y<=-1):
                role_a.y = world_y

        elif(action ==3):# right
            role_a.y+=1

            if(role_a.y>=world_y):
                role_a.y = 0
            
        elif(action ==4):
            pass
        else:
            raise(BaseException('wrong'))

    def render(role_a):
        if(role_a.game.show_game):
            s = role_a.game.screen
            render_x = role_a.x
            render_y = role_a.y

            s.blit(role_a.surface,(render_x,render_y))
        

    def reward_f(role_a,distance):
        if(role_a.flag==0):
            reward = distance
        else:
            # 如果是狼，则距离越大，越不好
            reward = 2**10 - 1 * distance
        role_a.reward = reward 



class main_a():

    def load(g,file_path):
        with open(file_path, "rb") as f:
        # with open('./model/cart_pole/a.pkl', "rb") as f:
            dqn = pickle.load(f)
        return dqn
    
    def save(g,dqn,file_path):
        with open(file_path, "wb") as f:
            pickle.dump(dqn, f)

    def distance_g(g,wolf,sheep):
        
        world_x = g.wold_x
        world_y = g.wold_y

        # bigx = 0
        if(wolf.x>sheep.x):
            bigx = wolf.x
            smallx = sheep.x
        else:
            bigx = sheep.x
            smallx = wolf.x
        
        # print('b s')
        # print(bigx,smallx)

        dis_x =abs(bigx-smallx)
        if(dis_x > world_x/2):
            smallx = smallx + world_x
            dis_x = smallx-bigx
        else:
            dis_x = abs(bigx-smallx)

        # print(smallx,bigx)

#y


        if(wolf.y>sheep.y):
            bigy = wolf.y
            smally = sheep.y
        else:
            bigy = sheep.y
            smally = wolf.y


        dis_y =abs(bigy-smally)
        if(dis_y > world_y/2):
            smally = smally + world_y
            dis_y = smally-bigy
        else:
            dis_y = abs(bigy-smally)

        distance = dis_x**2 + dis_y**2
        distance = distance**0.5

        
        return distance
    

            
    def judge(g,wolf,sheep):
        # sheep = g.sheep
        # wolf = 
        distance= g.distance_g(wolf,sheep)
        # print(distance)
        if(distance<20):
            g.terminate = 1
        else:
            g.terminate = 0

        
    def init(g):
        sheep = g.sheep
        
        sheep.x = 333
        sheep.y = 333
        # sheep.x = random.randint(0,g.wold_x-1)
        
        # sheep.y = random.randint(0,g.wold_y-1)


        for i,ele in enumerate(g.wolf_list):
        
            ele.x = random.randint(0,774-1) #774 #
            ele.y = random.randint(0,774-1) #774 #
            ele.integral = 0


    def main(g):
        show_game = 0
        init_train = 1
        plt_on = 0  # 是否显示图表



        if(init_train):
            print('init_train')


            mlp_architecture = [4,'wefwoefjo',4]
            wolf_dqn = dqn.dqn(mlp_architecture)
            wolf_dqn.episode = 0
            
            wolf_dqn.epsilon =  0.9
        
        else:
            file_path = './model/128_0.26373954491059526.pkl'
            wolf_dqn = g.load(file_path)
        print('train',wolf_dqn.epsilon)

        episode = wolf_dqn.episode
        # g.epsilon    = wolf_dqn.epsilon

        g.show_game = show_game
            

        if(plt_on ==1):
            plt.ion()
            plt.figure(1)
            t_list = list()
            # result_list=list()

            t_list = list()
            
            x_list = list()
            xd_list = list()
            th_list = list()
            thd_list = list()


        g.wold_x = 777
        g.wold_y = 777


        if(show_game):
            #使用pygame之前必须初始化
            pygame.init()
            #设置主屏窗口 ；设置全屏格式：flags=pygame.FULLSCREEN
            screen = pygame.display.set_mode((g.wold_x,g.wold_y))
            #设置窗口标题
            pygame.display.set_caption('wolf_sheep_battle')
        
        else:
            screen = 0
        g.screen = screen

        g.terminate = 0

        wolf_color = 'black'


        wolf_list = list()


        
        wolf_quantity = 1
        for i in range(wolf_quantity):
            ww  = role(wolf_color)
            ww.flag = 1
            wolf_list.append(ww)
            ww.dqn = wolf_dqn

        
        sheep = role('white')
        sheep.flag = 0

        rl = list()
        rl.append(sheep)
        rl+=wolf_list
        g.rl = rl


        for i,ele in enumerate(rl):
            ele.game = g
            ele.screen = ele.game.screen

        g.wolf_list = wolf_list
        g.sheep = sheep
        xii = 0
        
        while(1):
            g.init()

            # sheep.x = 3
            # sheep.y = 3
            step = 0
            while (1):
                if(show_game):
                    screen.fill('yellowgreen')
                    for event in pygame.event.get():    # 关闭游戏
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()


                for i,ele in enumerate(wolf_list):
                    
                    
                    ele.old_d = g.distance_g(ele,sheep)
                    ele.view = [ele.x,ele.y,sheep.x,sheep.y]
                    
                    ele.move()

                    
                    ele.new_d = g.distance_g(ele,sheep)

                    ele.view_after_action = [ele.x,ele.y,sheep.x,sheep.y]

                    # 计算奖励，并将数据写入经验内存

                    # distance = g.distance_f()
                    # ele.reward_f(  distance )
                    
                    
                    ####
                    view = torch.tensor(ele.view).reshape((1,4))
                    action = ele.action
                    # action = torch.tensor(ele.action).reshape((1,1))
                    view_after_action = torch.tensor(ele.view_after_action).reshape((1,4))
                    # reward = torch.tensor(ele.reward).reshape((1,1))
                    
                    ele.integral = 0.9*ele.integral + ele.new_d

                    reward_list = list()

                    # reward_list.append(10)
                    rr = 100*(ele.old_d-ele.new_d)
                    # if(rr<=0):rr = 0

                    
                    # print(ele.old_d)
                    # print(ele.new_d)
                    # print(rr)
                    # print('\n\n')


                    reward_list.append(rr)
                    # reward_list.append((-0.1)* ele.integral )
                    # reward_list.append((-1)* ele.new_d)
                    
                    reward = 0
                    for ir,rr in enumerate(reward_list):
                        
                        reward = reward + rr 
                    #     print(ir,rr)
                    # print(reward)
                    # print('\n\n')


                    # experience = [view,action,view_after_action,evaluate]
                    # print('ex',experience)
                    view = view.reshape((4))
                    view = np.array(view)
                    view_after_action = view_after_action.reshape((4))

                    view_after_action = np.array(view_after_action)
                    # print(view)
                    ele.dqn.store_transition(view,action,reward,view_after_action)
                    # print('ev',ele.reward)

                for i,ele in enumerate(rl):
                    ele.render()
                for i,ele in enumerate(wolf_list):

                    g.judge(ele,sheep)

                step+=1
                if(g.terminate==1):
                    print('wolf_win',episode)
                    xjj = 1
                    break
                else:
                    if(step>2**12):
                        # print('sheep_win',episode)
                        xjj = 0
                        break



                if(g.show_game):
                    if(episode>2**7):
                        # time.sleep(10**-3)
                        pass

                    pygame.display.flip() #更新屏幕内容
            # print('game_over')
            # print('train')
        
            # for i,ele in enumerate(rl):
            #     ele.dqn.learn()
            # xjj=1
            xii = 0.9* xii +xjj

            episode +=1

            if(episode%2**2==0):
                print(episode,xii,xjj)
                if(plt_on):
                    t_list.append(episode)
                    x_list.append(xii)
                    
                    plt.plot(t_list,x_list,c='red')
                    # print(float(fl))

                    plt.pause(0.001)
            xjj = 0

            # print(episode)

            wolf_dqn.episode = episode 
            if(episode % 2**7==0):
                file_path = './model/'+str(episode)+'_'+str(xii)+'.pkl'
                g.save(wolf_dqn,file_path)
                print('save',file_path)


            # print('ggg',g.distance_f())
            # for ii in range(1):
            wolf_dqn.learn()


if __name__ == "__main__":
    main_a().main()