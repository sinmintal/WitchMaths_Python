import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def func(theta, t):
    return theta


def func1(theta, t):
    return -32.17 * np.sin(theta)


def func2(theta, t):
    return -32.17 * theta


def RungeKuttaMethod(f, theta, aux, t):
    """
    θ'' + g * sin θ = 0
    θ'' + g * θ = 0

    θ(0) = pi/6
    θ'(0) = 0
    """
    h = t[1] - t[0]
    theta0 = math.pi / 6  # ф-ия в нуле
    thetaDev0 = 0  # производная ф-ии в нуле
    for i in range(len(t) - 1):
        theta.append(theta0)
        aux.append(thetaDev0)
        l1 = h * thetaDev0
        k1 = h * f(theta0, t)
        l2 = h * (thetaDev0 + 0.5 * k1)
        k2 = h * f(theta0 + 0.5 * l1, t + 0.5 * h)
        l3 = h * (thetaDev0 + 0.5 * k2)
        k3 = h * f(theta0 + 0.5 * l2, t + 0.5 * h)
        l4 = h * (thetaDev0 + k3)
        k4 = h * f(theta0 + l3, t + h)

        theta0 += (l1 + 2 * l2 + 2 * l3 + l4) / 6.
        thetaDev0 += (k1 + 2 * k2 + 2 * k3 + k4) / 6.


t = np.linspace(0, 8, 300)
theta = []  # ф-ия
aux = []  # вспомогательная ф-ия перехода для учёта первой произв.
fig, (frst, scnd) = plt.subplots(1, 2, figsize=(10, 3))
RungeKuttaMethod(func1, theta, aux, t)
frst.plot(theta, aux, color="red")
frst.grid(True)
RungeKuttaMethod(func2, theta, aux, t)
scnd.plot(theta, aux, color="blue")
scnd.grid(True)
plt.tight_layout()
plt.show()

# ОТРИСОВКА ОБЫЧНОЙ Ф-ИИ И Ф-ИИ ПРОИЗВОДНОЙ
# придать значения ф-ий: func и dev
# plt.plot(t, theta)
# plt.plot(t, aux)
# plt.show()