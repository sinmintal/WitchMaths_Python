import numpy as np
import matplotlib.pyplot as plt
from array import *

# объявление интервала [a, b] (область ф-ий) и N - кол-во вычисляемых значений

N = np.array([3, 4, 8, 10, 16, 3, 64, 256], dtype=int)
A = array('f', [2, 0.5, 4, 0])
B = array('f', [4, 3, 10, 12])

""" Фсякие функции
y = x^3 - 6.5 * x^2 + 11 * x + 11 * x - 4
y = 3 * cos (pi * x / 8)
y = e^(- x / 4 ) * sin (x / 3)
y = 8 * x * e^(- x^2 / 12)
"""


def func1(x):
    return x ** 3 - 6.5 * x ** 2 + 11 * x - 4


def func2(x):
    return 3 * np.cos(np.pi * x / 8)


def func3(x):
    return np.exp(-x / 4) * np.sin(x / 3)


def func4(x):
    return 8 * x * np.exp(-x ** 2 / 12)


# Вычисление полинома Лагранжа
def largange_moment(x, y, x0):
    rez = 0
    for j in range(len(y)):
        num = 1  # числитель
        den = 1  # знаменатель
        for i in range(len(x)):
            if i == j:
                num = num * 1  #
                den = den * 1  #
            else:
                num = num * (x0 - x[i])
                den = den * (x[j] - x[i])
        rez = rez + y[j] * num / den
    return rez


# Вычисление полинома Ньютона
def newton_moment(x, y, x0):
    residuals = np.zeros((len(x), len(x)))  # Для невязки также как и в алгоритме на джаве размер n x n
    rez = y[0]
    for i in range(0, len(x)):
        residuals[i, 0] = y[i]  # заполняем y для вычисления по формуле
    temp_sum = 1.0
    for i in range(1, len(x)):
        temp_sum = temp_sum * (x0 - x[i - 1])
        for j in range(i, len(x)):
            residuals[j, i] = (residuals[j, i - 1] - residuals[j - 1, i - 1]) / (x[j] - x[j - i])
        rez += temp_sum * residuals[i, i]
    return rez


# Заполнение массива x
def define_x(a, b, n):
    # x = np.zeros((1, n), dtype=np.float) TODO: посмотреть что работает лучше, empty или np.zeros
    h = (b - a) / n
    for i in range(n):
        x = np.arange(a, b, h)
    # print(x)
    return x


# организация данных для полиноминальных графиков
def fillDataGrafics(x, y):
    polynomial_x = np.linspace(np.min(x), np.max(x), 100)  #

    polynomial_y = [largange_moment(x, y, i) for i in polynomial_x]
    title = "LAGRANGE"
    with open(r"Lagrange.txt", "w") as file:
        file.write("[Интерполяция Лагранжа]")
        for pol_x in polynomial_x:
            file.write(format(pol_x) + '\n')
    showGrafics(title, x, y, polynomial_x, polynomial_y)

    polynomial_y = [newton_moment(x, y, i) for i in polynomial_x]
    title = "NEWTON"
    with open(r"Newton.txt", "w") as file:
        file.write("[Интерполяция Ньютона]")
        for pol_y in polynomial_y:
            file.write(format(pol_y) + '\n')
    showGrafics(title, x, y, polynomial_x, polynomial_y)


# отрисовка графиков
def showGrafics(string, x, y, polynomial_x, polynomial_y):
    plt.plot(y, ':', color="purple")
    if string == "NEWTON":
        plt.title(string + "'s Interpolation")
        plt.plot(x, y, 'o', label='steel', color="orange")
        plt.plot(polynomial_x, polynomial_y, 'r', color="red")
    else:
        plt.title(string + "'s Interpolation")
        plt.plot(x, y, 'o', color="green")
        plt.plot(polynomial_x, polynomial_y, 'r', color="blue")

    plt.legend(['f(x)', 'x/y from table', 'Interpolation res'])
    plt.grid(True)
    plt.show()


# Объявление переменных
n = N[4]  # можно записывать резы по 8 разным числам в разные файлы, иначе тут ад
x = np.empty(n)
y = np.empty(n)

f = open("results.txt", "w")
for i in range(0, len(A)):
    x = define_x(A[i], B[i], n)
    y = func1(x)  # Можно выбирать ф-ию
    for j in range(n):
        f.write("Для {}".format(x[j]) +
                " в интервале [{}".format(A[i]) +
                ", {}".format(B[i]) +
                "] значение ф-ии равно {}".format(y[j]) +
                "\n")
    f.write("\n")
    fillDataGrafics(x, y)

f.close()

print("Найти значение в точке вне сетки:\n")
print(newton_moment(x, y, -10))
print(largange_moment(x, y, -10))
print(func1(-10), "\n")
