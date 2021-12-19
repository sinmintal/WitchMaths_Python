import math
import matplotlib.pyplot as plt
import numpy as np
import copy


def integral(lPairs, dx):
    E = 0
    for pair in lPairs:
        E += dx * (pair[0] ** 2 + pair[1] ** 2)
    return E


fig = plt.figure()
plt.ion()
# u (x, t)
ax = fig.add_subplot(3, 2, 1, ylim=(-1.1, 1.1), xlim=(-0.5, 1.05), title='u(x, t), x = [0:1]', xlabel='x', ylabel='u')
# интеграл
ax2 = fig.add_subplot(3, 2, 2, title='E(t)', xlabel='t', ylabel='E')
# поверхность
ax3 = fig.add_subplot(3, 2, (3, 6), projection='3d', title='u(x, t)', xlabel='x', ylabel='t', zlabel='u')
# ax.minorticks_on()  # для коорд. сеток
# ax2.minorticks_on()
ax.grid(which='major', color=(0.5, 0.5, 0.5), linewidth=1)
ax.grid(which='minor', color=(0.5, 0.5, 0.5), linestyle=':')
ax2.grid(which='major', linewidth=1)
ax2.grid(which='minor', color=(0.5, 0.5, 0.5), linestyle=':')  # стили

line1, = ax.plot([], [])
line2, = ax2.plot([], [])

h = 0.05
tau = 0.5 * np.power(h, 1)
# x = np.linspace(0, 1+h)
x = np.arange(0, 1 + h, h)
# u позапредыдущее значение
ubl = [*map((lambda x: np.sin(np.pi * x)), x)]
ut = [*map((lambda x: 16 * x ** 3 * (1 - x) ** 2), x)]
uder = [[]]  # производные

for i in range(0, len(ubl) - 1):
    # ut и ux
    uder[0].append([(ut[i + 1] - ut[i]) / h, (ubl[i + 1] - ubl[i]) / h])
uder.append([])  # добавление вектора
# u предыдущее
ul = len(ubl) * [0]  # можно заменить на np.zeros
TH = tau ** 2 / h ** 2  # переменная для хранения t/h
for j in range(1, len(ubl) - 1):
    power = - ((j * h) ** 2) / 2
    a = TH * ((j * h) * np.exp(power))  # uj+1
    b = (2 - 2 * TH) * ((j * h) * np.exp(power))  # ujn
    c = TH * ((j * h) * np.exp(power))  # uj-1n
    ul[j] = b * ubl[j] + (a * ubl[j + 1] + c * ubl[j - 1]) / 2 - tau * ut[j]

for i in range(0, len(ubl) - 1):
    # производная предыдущее - позапредыдущее
    # попробовать с ut
    uder[1].append([(ul[i] - ubl[i]) / tau, (ubl[i + 1] - ubl[i]) / h])
uder.append([])
# интеграл
Et = [integral(uder[0], h), integral(uder[1], h)]
uderInd = 2  # индекс массива для производной

unow = len(ul) * [0]  # текущее значение ubl
# матрица для всхе значений u
mu = [ubl, ul]
t = 2 * tau
# основной расчет
while t <= 5:
    for j in range(1, len(ubl) - 1):
        power = - ((j * h) ** 2) / 2
        a = TH * ((j * h) * np.exp(power))  # uj+1
        b = (2 - 2 * TH) * ((j * h) * np.exp(power))  # ujn
        c = TH * ((j * h) * np.exp(power))  # uj-1n
        unow[j] = (a * ul[j + 1] + b * ul[j] + c * ul[j - 1]) - ubl[j]

    # обновление переменных
    ubl = copy.deepcopy(ul)
    ul = copy.deepcopy(unow)
    mu.append(copy.deepcopy(unow))
    t += tau

    for i in range(0, len(ubl) - 1):
        uder[uderInd].append([(ul[i] - ubl[i]) / tau, (ubl[i + 1] - ubl[i]) / h])
    uder.append([])
    Et.append(integral(uder[uderInd], h))
    # uderInd++
    uderInd += 1

    line1.set_data(x, unow)
    ax.relim()
    lt = list(np.arange(0, t, tau))  # текущее кол-во стов
    if len(Et) == len(lt):
        line2.set_data(lt, Et)
        ax2.relim()
        ax2.autoscale_view(True, True, True)
    plt.show()

x = (np.arange(0, 1.05, h))
t = (np.arange(0, 5, tau))
# x = np.linspace(0, 1, 50)
# t = np.linspace(0, 5, 50)
x, t = np.meshgrid(x, t)
u = np.array(mu)  # массив
# flatten - упрощает до 1D массива, оч нужно для отображения без ошибок
ax3.plot_trisurf(x.flatten(), t.flatten(), u.flatten(), linewidth=10, cmap='inferno')
# surf = ax3.plot_surface(x, t, u, rstride=1, cstride=1, cmap='viridis')
# surf = ax3.plot_trisurf(x, t, u, linewidth=0.5, cmap='inferno') # rstride=1, cstride=1,
# ax3.set_xlim(-10, 10)
# ax3.set_ylim(-10, 10)
plt.pause(20)
