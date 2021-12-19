import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def EulerMethod(f, y0, t):
    y = np.zeros([len(t), len(y0)])
    y[0] = y0
    for n in range(0, len(t) - 1):
        y[n + 1] = y[n] + f(y[n], t[n]) * (t[1] - t[0])
    return y


def RungeKuttaMethod(f, y0, t):
    y = np.zeros([len(t), len(y0)])
    h = t[1] - t[0]
    y[0] = y0
    for i in range(0, len(t) - 1):
        k1 = f(y[i], t[i])
        k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(y[i] + k3, t[i] + h)
        y[i + 1] = y[i] + ((k1 + 2 * k2 + 2 * k3 + k4) / 6.)
    return y


def func(y, t):
    x1, x2 = y
    k1 = 3
    k2 = 2e-3
    k3 = 6e-4
    k4 = 0.5
    dxdt = [x1 * (k1 - k2 * x2), x2 * (-k4 + k3 * x1)]
    return np.array(dxdt)


x01 = x02 = 500
#t = np.linspace(0, 20, 60)
t = np.linspace(0, 20, 60)
y0 = [x01, x02]
# y = EulerMethod(func, y0, t)
y = RungeKuttaMethod(func, y0, t)
y_ = integrate.odeint(func, y0, t)
plt.figure()
plt.plot(t, y[:, 0], 'b-', color='red')
plt.plot(t, y[:, 1], 'b-', color='blue')
plt.grid(True)
plt.show()

plt.plot(t, y_[:, 0], 'b-', color='red')
plt.plot(t, y_[:, 1], 'b-', color='blue')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(y[:, 0], y[:, 1], '-')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(y_[:, 0], y_[:, 1], '-')
plt.grid(True)
plt.show()
