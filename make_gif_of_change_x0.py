import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.linalg import sqrtm
import matplotlib.animation as animation

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

def pdcomposeoringin(fk, num):
    dimension = np.shape(fk)
    I = np.identity(dimension[0])
    result = I
    fk2 = fk
    while num != 0:
        result = fk2 + result
        fk2 = np.dot(fk2, fk)
        num = num - 1
    return result

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

    exparray = sqrtm(exparray)/np.linalg.norm(sqrtm(exparray))
    return exparray

def createdata(dim):
    A = np.identity(dim)
    num = 1
    while num != dim + 1:
        A[num-1][num-1] = num
        num = num + 1
    print("数据生成成功", A)
    A = sqrtm(A) / np.linalg.norm(sqrtm(A))
    return A

def pdcompose(fk, num):
    dimension = np.shape(fk)
    iden = np.identity(dimension[0])
    if num == 1:
        result = fk + iden
        return result
    if num == 2:
        result = (fk @ fk) + fk + iden
        return result
    if num % 2 == 0:
        print("num1", num)
        fk2 = fk @ fk
        result = pdcompose(fk2, (num-2)/2) @ (fk2 + fk) + iden
        return result
    else:
        print("num2", num)
        fk2 = fk @ fk
        result = pdcompose(fk2, (num-1)/2) @ (fk + iden)
        return result

def tasuokain(A, p, x0):
    literation = 0
    residualnorm = []
    axixx = []
    dimension = np.shape(A)
    I = np.identity(dimension[0])
    X = I * x0
    # print("X", X)
    Xtest = fastPow(X, p)
    # L2 = np.linalg.norm(A - Xtest) / np.linalg.norm(A)     #残差
    L2 = np.linalg.norm(A - Xtest) #残差

    if L2 == 0:
        residualnorm.append(0)
        axixx.append(literation)
        result = X
        return (axixx, residualnorm, result, literation)

    L2 = math.log(L2, 10)
    residualnorm.append(L2)
    axixx.append(literation)
    literation = literation + 1
    H = (A - I)/p
    # print("H",H)
    # print("A", A)
    # print("A-I", A-I)

    p1 = p-1                #为了使用p-1而不改变p的值
    # L2save = 0

    while 1:
        if L2 < -12:
            break
        if literation > 5:
            if residualnorm[literation-1]==residualnorm[literation-2]==residualnorm[literation-3]==residualnorm[literation-4]:
                break
        X1 = X + H
        result = X1
        F = X @ np.linalg.inv(X1)
        presult = pdcompose(F, p-2)
        H = (-1)/p * ((-p1 * F + p * I) @ presult - p1 * I) @ H
        X = X1
        Xtest = fastPow(result, p)
        L2 = np.linalg.norm(A - Xtest)  # 残差
        L2 = math.log(L2, 10)
        residualnorm.append(L2)
        literation = literation + 1
        axixx.append(literation)

    return (axixx, residualnorm, result, literation)


literation = []
x0_result_norm = []
x0_result_norm2 = []
A = [[20,9],
     [0,15]]
A = np.array(A)
dimention = np.shape(A)
I = np.identity(dimention[0])
B = [[2.11474253,0.26415514],
     [0,1.96798967]]
B = np.array(B)
p = 4 #观察初期値x0变化的结果时的p
x0 = [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5, 1.8, 1.9, 2] #观察初期値x0变化时的x0
y = [] #存储迭代次数
z = [] #存储相对残差
axixx = []
len = len(x0)    #注意切换观察对象时len要改
num = 0
while num != len:
    result = tasuokain(A, p, x0[num])
    y.append(result[3])
    z.append(result[1])
    axixx.append(result[0])
    literation.append(result[3])
    x0_result_norm.append(np.linalg.norm(I*x0[num]-B))
    x0_result_norm2.append(np.linalg.norm(fastPow(x0[num] * I, p) - A))
    num = num + 1


# gif
fig, ax= plt.subplots(figsize=(8,8))
def animate(i):
    ax.cla()
    ax.set_xlim(0,20,5)
    ax.set_xlabel("iteration", fontsize = 24)
    ax.set_ylim(-15,10,1)
    ax.set_ylabel("residual norm(log10)", fontsize=24)
    ax.tick_params(labelsize = 24)
    ax.plot(axixx[i], z[i],"o-")
    ax.set_title(f'x0={x0[i]}*I', fontsize=24)
ani = animation.FuncAnimation(fig = fig,
                              func=animate,
                              frames=len,
                              # init_func=init,
                              interval=1000,
                              blit=False)
plt.show()
ani.save('change_x01.gif')
print(x0_result_norm,literation)
print("wwwww",x0_result_norm2)

