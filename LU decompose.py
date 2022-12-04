import numpy as np
import matplotlib.pyplot as plt
import time

# Алгоритм LU разложения

def LU(n):

    startTime = time.time()

    a = np.random.randint(low=1, high=100, size=(n,n))
    for k in range(n):
        a[k, k] *= 30
    lu_matrix = np.matrix(np.zeros([a.shape[0], a.shape[1]]))
    n = a.shape[0]

    for k in range(n):
        for j in range(k, n):
            lu_matrix[k, j] = a[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
        for i in range(k + 1, n):
            lu_matrix[i, k] = (a[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]
    # return lu_matrix

    L = get_L(lu_matrix)
    U = get_U(lu_matrix)

    # print('Матрица L')
    # print(L)
    # print('Матрица U')
    # print(U)
    # print('Проверка')
    # print(L * U)

    return(time.time() - startTime)

# Получение L

def get_L(m):
    L = m.copy()
    for i in range(L.shape[0]):
            L[i, i] = 1
            L[i, i+1 :] = 0
    return np.matrix(L)

# Получение U

def get_U(m):
    U = m.copy()
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return U

# Временная сложнось

def tests(N):
    sizes=[]
    times=[]
    avg=0
    for j in range(3,N):
        avg+=LU(j)
        if (j % 25==0):
            sizes.append(j)
            times.append(((avg)/10)*1000000)
            avg=0
    return (sizes,times)

count=10
s_arr=[]
t_arr=[]
for j in range(1,count):
    s, t = tests(100)
    s_arr.append(s)
    t_arr.append(t)
s=np.array(s_arr[0])
t=np.array(t_arr[0])
for j in range(1,len(s_arr)):
    s+=np.array(s_arr[j])
    t+=np.array(t_arr[j])
s=s/count
t=t/count

# Визуализация зависимости времени от размерности 

plt.title("Зависимость времени от размерности матрицы")
plt.xlabel("Размерность матрицы")
plt.ylabel("Время выполнения алгоритма в мкс")
plt.plot(s,t)
plt.show()

#Тестирование при удвоении входных данных

count=5
s=[5000, 10000, 20000]
avg_arr=[]

for l in s:
    avg=0
    for j in range(count):
        avg+=LU(l)
    avg_arr.append(avg/count)

print(avg_arr)