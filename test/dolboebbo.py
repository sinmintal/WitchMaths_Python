import math
import matplotlib.pyplot as plt
import numpy as np
import copy


def fu(x):
    return 16 * x ** 3 * (1 - x) ** 2


def integral(lpairs, dx):
    E = 0
    for pair in lpairs:
        # ut^2 + ux^2 с учетом того, что там произведнеие суммы
        E += (pair[0] ** 2 + pair[1] ** 2) * dx
    return E


def coefficients(u, u0, u_, var, h):
    for j in range(1, len(u_) - 1):
        a = (- tau ** 2 / h ** 2) * ((j * h) * np.exp(- (j * h) ** 2 / 2))
        b = (1 + 2 * tau ** 2 / h ** 2) * ((j * h) * np.exp(- (j * h) ** 2 / 2))
        c = (-tau ** 2 / h ** 2) * ((j * h) * np.exp(- (j * h) ** 2 / 2))
        # a = (tau ** 2 / h) * (np.cos(j * h) / h - np.sin(j * h) + 2)
        # b = tau ** 2 * (-(2 * np.cos(j * h)) / h ** 2 + (np.sin(j * h) - 2) / h - np.sin(np.pi * j * h)) + 2
        # c = tau ** 2 * np.cos(j * h) / h ** 2
        # a = (tau ** 2 / h) * (math.exp(-(j * h) ** 2 / 2) / h + 4)
        # b = tau ** 2 * (-(4 * math.exp(-(j * h) ** 2 / 2)) / h ** 2 + 2)
        # c = tau ** 2 * (j * h) * math.exp(-(j * h) ** 2 / 2) / h ** 2
        if var == 0:
            u[j] = (a * u0[j + 1] + b * u0[j] + c * u0[j - 1]) / 2 - u_[j]
        else:
            u[j] = (a * u0[j + 1] + b * u0[j] + c * u0[j - 1]) - u_[j]


""" ПОДГОТОВКА ГРАФИКОВ К ОТРИСОВКЕ"""
fig = plt.figure()
plt.ion()
ax = fig.add_subplot(3, 2, 1, ylim=(-1.1, 1.1), xlim=(-0.5, 1.05), title='u(x, t), x = [0:1]', xlabel='x', ylabel='u')
ax2 = fig.add_subplot(3, 2, 2, title='E(t)', xlabel='t', ylabel='E')
ax3 = fig.add_subplot(3, 2, (3, 6), projection='3d', title='u(x, t)', xlabel='x', ylabel='t', zlabel='u')
ax.minorticks_on()
ax2.minorticks_on()
ax.grid(which='major', linewidth=1)
ax.grid(which='minor', linestyle=':')
ax2.grid(which='major', linewidth=1)
ax2.grid(which='minor', linestyle=':')

line1, = ax.plot([], [])
line2, = ax2.plot([], [])

h = 0.045
alpha = 1
tau = 0.5 * np.power(h, alpha)
x = np.arange(0, 1 + h, h)

ubl = [*map(lambda x: np.sin(math.pi * x), x)]  # ux0
ul = [*map(lambda x: 16 * np.power(x, 3) * np.power(1 - x, 2), x)]  # utx0
uder = [[]]

coefficients(ul, ubl, ubl, 0, h)

for i in range(0, len(ubl) - 1):
    uder[0].append([(ul[i+1] - ul[i]) / 2 * tau, (ubl[i+1] - ubl[i] / h)])
    uder.append([])
    uder[1].append([(ul[i+1] - ul[i]) / 2 * tau, (ubl[i + 1] - ubl[i]) / h])
uder.append([])

Et = [integral(uder[0], h), integral(uder[1], h)]  # (ut, ux)
uderInd = 2
unow = np.zeros(len(ul))
mu = [ubl, ul]
t = tau + tau

while t <= 5:
    coefficients(unow, ul, ubl, 1, h)

    ubl = copy.deepcopy(ul)
    ul = copy.deepcopy(unow)
    mu.append(copy.deepcopy(unow))
    t += tau

    for i in range(0, len(ubl) - 1):
        uder[uderInd].append([(ul[i] - ubl[i]) / tau, (ubl[i + 1] - ubl[i]) / h])
    uder.append([])
    Et.append(integral(uder[uderInd], h))
    uderInd += 1

    line1.set_data(x, unow)
    ax.relim()
    lt = list(np.arange(0, t, tau))
    if len(Et) == len(lt):
        line2.set_data(lt, Et)
        ax2.relim()
        ax2.autoscale_view(True, True, True)
    plt.show()

x = np.arange(0, 1 + h, h)
t = np.arange(0, 5, tau)
x, t = np.meshgrid(x, t)
u = np.array(mu)
surf = ax3.plot_surface(x, t, u, rstride=1, cstride=1, cmap='viridis')
plt.pause(25)
