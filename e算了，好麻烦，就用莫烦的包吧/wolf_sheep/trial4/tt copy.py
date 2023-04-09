

# 研究下 那个收敛函数



i = 0
k = 0.9
ep = 0
while (1):
    j = 1
    i = k*i +(1-k)*j
    print(i)
    if(i>0.999999):
        print(ep)
        break
    ep+=1
    # print(1/k)
