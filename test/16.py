import matplotlib.pyplot as plt
import numpy as np
import copy


def integral(lpairs, dx):
    E = 0
    for pair in lpairs:
        # ut^2 + ux^2 с учетом того, что там произведнеие суммы
        E += (pair[1] ** 2 + pair[0] ** 2) * dx
    return E


def coefficients(u, u0, u_, h):
    for j in range(1, len(u_) - 1):
        a = (tau ** 2 / h ** 2) * ((j * h) * np.exp(- (j * h) ** 2 / 2) + 4) * u_[j]
        b = (1 - 2 * tau ** 2 / h ** 2) * ((j * h) * np.exp(- (j * h) ** 2 / 2)) * u_[j]
        c = (tau ** 2 / h ** 2) * ((j * h) * np.exp(- (j * h) ** 2 / 2)) * u_[j]
        u[j] = (a * u0[j + 1] + b * u0[j] + c * u0[j - 1]) - u_[j]


fig = plt.figure()
plt.ion()
ax = fig.add_subplot(3, 2, 1, ylim=(-1.1, 1.1), xlim=(-0.5, 1.05), title='u(x, t), x = [0:1]', xlabel='x', ylabel='u')
ax2 = fig.add_subplot(3, 2, 2, title='E(t)', xlabel='t', ylabel='E')
ax3 = fig.add_subplot(3, 2, (3, 6), projection='3d', title='u(x, t)', xlabel='x', ylabel='t', zlabel='u')
ax.minorticks_on()
ax2.minorticks_on()
ax.grid(which='major', color=(0.5, 0.5, 0.5), linewidth=1)
ax.grid(which='minor', color=(0.5, 0.5, 0.5), linestyle=':')
ax2.grid(which='major', linewidth=1)
ax2.grid(which='minor', color=(0.5, 0.5, 0.5), linestyle=':')

line1, = ax.plot([], [])
line2, = ax2.plot([], [])

h = 0.03
alpha = 1.25
tau = 0.5 * np.power(h, alpha)
x = np.arange(0, 1 + h, h)
ubl = [*map(lambda x: np.sin(np.pi * x), x)]
ubl[0], ubl[-1], uder = 0, 0, [[]]

for i in range(0, len(ubl) - 1):
    uder[0].append([*map(lambda x: 16 * x ** 3 * (1 - x) ** 2, x)])
uder.append([])

ul = len(ubl) * [0]

coefficients(ul, ubl, ubl, h)
for i in range(0, len(ubl) - 1):
    uder[1].append([(ul[i] - ubl[i]) / tau, (ubl[i + 1] - ubl[i]) / h])

uder.append([])
Et = [integral(uder[0], h), integral(uder[1], h)]   # (ut, ux)
uderInd = 2
unow = len(ul) * [0]
mu = [ubl, ul]
t = 0 + tau + tau
while t <= 5:
    coefficients(unow, ul, ubl, h)
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
    # plt.pause(0.001)

x = np.arange(0, 1 + h, h)
t = np.arange(0, 5, tau)
x, t = np.meshgrid(x, t)
u = np.array(mu)
surf = ax3.plot_surface(x, t, u)
plt.pause(25)
