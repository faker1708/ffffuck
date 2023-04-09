

# 研究下 那个收敛函数


# 需求，近似表达定积分



k = 2**-4   # 控制积分窗口

i = 0
ep = 0
while (1):
    j = 1
    i = (1-k)*i +(k)*j
    print(i)
    
    if(1-i<k):
        print(ep)
        break
    ep+=1
    # print(1/k)
