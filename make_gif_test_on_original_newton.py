import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.linalg import sqrtm
import matplotlib.animation as animation

def creattestdata():
    exparray = np.zeros(shape=(1440, 1440))      #更换
    path = "F:/名古屋大学学习相关/文献紹介/msc01440.mtx"   #更换
    data = open(path, mode="r", encoding="iso8859-1")
    i = 14   #更换
    # exparray = np.zeros(shape=(1374, 1374))  # 更换
    # path = "F:/名古屋大学学习相关/文献紹介/nnc1374.mtx"  # 更换
    # data = open(path, mode="r", encoding="iso8859-1")
    # i = 2  # 更换

    count = 0
    while i != 0:
        data.__next__()
        i = i - 1
    while True:
        try:
            string = data.readline().split()
            try:
                column = int(string[0])
                row = int(string[1])
                number = float(string[2])
                count = count + 1
                exparray[column-1, row-1] = number
            except:
                print("装填完毕")
                break
        except StopIteration:
            print("循环结束")
            break
    return exparray

def fastPow(A, p):  # x是底n是幂
    if p == 0:
        shape = np.shape(A)
        iden = np.identity(shape[0])
        return iden
    elif p < 0:
        print("n<0")
    elif p % 2 == 1:
        save = fastPow(A, (p - 1) / 2)
        result = save @ save @ A
        return result
    else:
        save = fastPow(A, p / 2)
        result = save @ save
        return result

def CaculateResidual(X, A, p):
    XP = fastPow(X, p)
    # print(XP)
    # L2 = np.linalg.norm(A - XP) / np.linalg.norm(A) #相対残差
    L2 = np.linalg.norm(A - XP) #残差
    if L2 == 0:
        return L2
    L2 = math.log(L2, 10)
    return L2


def Newton(A, p, x0):
    axixx = []
    axixy = []
    result = []
    num = 0
    dimension = np.shape(A)
    I = np.identity(dimension[0])
    X = I * x0
    residual = CaculateResidual(X, A, p)
    # print(residual)
    axixx.append(num)
    axixy.append(residual)
    result.append(X)

    while residual > -12 and residual != 0:
        num = num + 1
        X = ((p-1)*X + A@np.linalg.inv(fastPow(X, p-1)))/p
        residual = CaculateResidual(X, A, p)
        # print("residual", residual)
        # print("x",X)
        axixx.append(num)
        axixy.append(residual)
        result.append(X)
        if num == 30:
            break
    literation = num

    return axixx, axixy, result, literation

A = [[20,9],
     [0,15]]
B = [[2.11474253,0.26415514],
     [0,1.96798967]]
B = np.array(B)
# A = creattestdata()
A = np.array(A)
p = 4 #观察初期値x0变化的结果时的p
x0 = [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5, 1.8, 1.9, 2] #观察初期値x0变化时的x0
dimention = np.shape(A)
I = np.identity(dimention[0])

literation = []
z = [] #存储残差
x0_result_norm = []
x0_result_norm2 = []
axixx = []
len = len(x0)    #注意切换观察对象时len要改
num = 0
while num != len:
    result = Newton(A, p, x0[num])
    z.append(result[1])
    x0_result_norm.append(np.linalg.norm(x0[num]*I - B))
    x0_result_norm2.append(np.linalg.norm(fastPow(x0[num]*I,p)-A))
    axixx.append(result[0])
    literation.append(result[3])
    num = num + 1

fig, ax = plt.subplots(figsize=(8,8))

def animate(i):
    ax.cla()
    ax.set_xlim(-1,20)
    ax.set_xlabel("iteration",fontsize = 24)
    ax.set_ylim(-15,10)
    ax.set_ylabel("residual norm(log10)", fontsize = 24)
    ax.set_title(f'x0={x0[i]}*I',fontsize = 24)
    ax.tick_params(labelsize= 24)
    ax.plot(axixx[i], z[i], "o-")

ani = animation.FuncAnimation(fig=fig, func=animate, interval=2000, frames=len)
plt.show()

ani.save("make_gif_of_change_x0_Newton.gif")
print(x0_result_norm,literation)
print("wwwww",x0_result_norm2)


