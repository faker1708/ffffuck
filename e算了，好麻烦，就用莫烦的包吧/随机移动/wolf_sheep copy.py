import sys
import pygame
import time
import random
import dqn


# 多人竞技
# 非零和，与环境 变化现在还没有。
# 现在是最简单的版本，一只羊，一头狼，没有技能 ，速度相同 。感觉羊稳赢。


class role():

    def __init__(role_a,color='blue'):
        rs = pygame.Surface((2**4,2**4),flags=pygame.HWSURFACE)
        rs.fill(color)
        role_a.x = random.randint(0,777-1) #774 #
        role_a.y =  random.randint(0,777-1) #774 #
        role_a.surface = rs
        
        # role_a.screen = screen

    def move(role_a):
        # 
        # role_a.x +=1
        action = role_a.dqn.action_f('nouse')
        # print(action)
        
        world_x = role_a.game.wold_x
        world_y = role_a.game.wold_y

        # 循环位移 ，遇到边界就传送到另一边
        # 有限，无边界
        if(action ==0):# up
            role_a.x -=1
            
            if(role_a.x<=-1):
                role_a.x = world_x


        elif(action ==1):#down
            role_a.x+=1
            
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
            



    def render(role_a):
        s = role_a.game.screen
        s.blit(role_a.surface,(role_a.x,role_a.y))




class main_a():

    def distance_f(g):
        
        wolf = g.rl[0]
        sheep = g.rl[1]
        world_x = g.wold_x
        world_y = g.wold_y

        # bigx = 0
        if(wolf.x>sheep.x):
            bigx = wolf.x
            smallx = sheep.x
        else:
            bigx = sheep.x
            smallx = wolf.x
        
        dis_x =abs(bigx-smallx)
        if(dis_x > world_x/2):
            print('要反过来算') 
            smallx = smallx + world_x/2
            dis_x = smallx-bigx
        else:
            dis_x = abs(bigx-smallx)
#y
        if(wolf.y>sheep.y):
            bigy = wolf.y
            smally = sheep.y
        else:
            bigy = sheep.y
            smally = wolf.y


        dis_y =abs(bigy-smally)
        if(dis_y > world_y/2):
            print('要反过来算') 
            smally = smally + world_y/2
            dis_y = smally-bigy
        else:
            dis_y = abs(bigy-smally)

        distance = dis_x**2 + dis_y**2
        distance = distance**0.5
        return distance
    

            
    def judge(g):
        wolf = g.rl[0]
        sheep = g.rl[1]
        
        
        # print(wolf.x)
        # print(sheep.x)
        
        dx = abs(wolf.x-sheep.x)
        dy = abs(wolf.y-sheep.y)
        
        if(dx<20):
            if(dy<20):
                g.terminate = 1
        # else:
            # print('no')



    def main(g):
        

        g.wold_x = 777
        g.wold_y = 777


        #使用pygame之前必须初始化
        pygame.init()
        #设置主屏窗口 ；设置全屏格式：flags=pygame.FULLSCREEN
        screen = pygame.display.set_mode((g.wold_x,g.wold_y))
        #设置窗口标题
        pygame.display.set_caption('wolf_sheep_battle')
        # screen.fill('black')

        # plate_size = [777,777]

        g.terminate = 0
        g.screen = screen

        # world = 



        # for i in range(3):
        #     cc = role('yellow')
        #     # print(cc.x)
        #     cc.screen = screen


        #     rl.append(cc)
        wolf = role('darkgray')
        wolf.dqn = dqn.dqn()
        
        sheep = role('white')
        sheep.dqn = dqn.dqn()




        rl = list()
        rl.append(wolf)
        rl.append(sheep)
        g.rl = rl

        for i,ele in enumerate(rl):
            ele.game = g
            ele.screen = ele.game.screen
        

        while True:
            screen.fill('yellowgreen')
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


            for i,ele in enumerate(rl):
                ele.move()
                ele.render()

            g.judge()
            if(g.terminate==1):
                break

            time.sleep(10**-6)
            # time.sleep(10**0)

            pygame.display.flip() #更新屏幕内容
        print('game_over')



if __name__ == "__main__":
    main_a().main()