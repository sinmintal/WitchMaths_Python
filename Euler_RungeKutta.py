import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def f1(p, t):
    r = 0.1
    b = 0.02
    dpdt = r * b * (1 - p)
    return dpdt


def f2(v, t):
    m = 1.
    g = 9.8
    k = 0.002
    mvdt = - m * g - k * v * math.fabs(v)
    return mvdt


def f3(x, t):
    dxdt = 0.1 - ((3 * x) / (1000 + x))
    return dxdt


def f4(x, t):
    n1 = n2 = 2000
    n3 = 3000
    k = 6.22e-19
    dxdt = k * math.pow((n1 - (x / 2)), 2) * math.pow((n2 - (x / 2)), 2) * math.pow((n3 - (3 * x / 4)), 3)
    return dxdt


def f5(y, t):
    m = 100000
    k = 2 * 10 ** (-6)
    z = (m - y)
    dydt = k * z * y
    return dydt


def EulerMethod(f, y0, t):
    """ df - диф. функции: p'(t)/mv'/x'(t) и тд
        y0 - задача коши: p(0)/v(0)/x(0)
        t - время
    """
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        y[n + 1] = y[n] + f(y[n], t[n]) * (t[n + 1] - t[n])
    return y


def RungeKuttaMethod(f, y0, t):
    y = np.zeros(len(t))
    if (f == f4):
        h = 0.0009
    else:
        h = (20 / 30)
    y[0] = y0
    for i in range(0, len(t) - 1):
        k1 = f(y[i], t[i])
        k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(y[i] + k3, t[i] + h)
        y[i + 1] = y[i] + ((k1 + 2 * k2 + 2 * k3 + k4) / 6.)
    return y


def drawResults(t, yEu, yRK, y_):
    plt.plot(t, yEu, '.-', color="darkorange")
    plt.plot(t, yRK, '.-', color="blueviolet")
    plt.plot(t, y_, 'r-', color="forestgreen")
    plt.legend(['Метод Эйлера', 'Метод Рунге-Кутты', 'Встроенная ф-ия'])
    plt.grid()
    plt.show()


y0_array = [0.01, 8, 50, 0.0, 1000]
f_array = [f1, f2, f3, f4, f5]

for i in range(0, len(y0_array)):
    plt.title("График для " + str(i + 1) + "-й задачи")
    y0 = y0_array[i]
    f = f_array[i]
    if (i == 3):  # для 4-го пункта
        t = np.linspace(-0., 0.5, 200)
        yEu = EulerMethod(f, y0, t)
        yRK = RungeKuttaMethod(f, y0, t)
    else:
        t = np.linspace(0, 20, 30)  # массив значений 30 точек от 0 до 20
        yEu = EulerMethod(f, y0, t)
        yRK = RungeKuttaMethod(f, y0, t)
    y_ = odeint(f, y0, t)
    drawResults(t, yEu, yRK, y_)
