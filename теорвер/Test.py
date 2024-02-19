import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t

import numpy as np
import matplotlib.pyplot as plt

# Функции распределения и плотности распределения
def F(x):
    if x <= 0:
        return 0
    elif 0 < x <= 2:
        return x**2 / 16
    elif 2 < x <= 11/4:
        return x - 7/4
    elif x > 11/4:
        return 
    
# Производная по x
def f(x):
    if x <= 0:
        return 0
    elif 0 < x <= 2:
        return 1/8 * x
    elif 2 < x <= 11/4:
        return 1
    elif x > 11/4:
        return 0

# Значения x для построения графиков
x_values = np.linspace(-1, 3, 1000)

# График функции распределения F(x)
y_F = [F(x) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_F, label='F(x)')
plt.title('График функции распределения F(x)')
plt.xlim(-1,3)
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.show()

# График плотности распределения f(x)
y_f = [f(x) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_f, label='f(x)')
plt.title('График плотности распределения f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(-1,3)
plt.legend()
plt.show()

from scipy.integrate import quad

# Определенные интегралы для математического ожидания и дисперсии
def integrand_mean(x):
    return x * f(x)

def integrand_variance(x):
    return (x - quad(integrand_mean, -np.inf, np.inf)[0]) ** 2 * f(x)


# Вычисление математического ожидания
mean_value, _ = quad(integrand_mean, -np.inf, np.inf)

    # Вычисление дисперсии
variance_value, _ = quad(integrand_variance, -np.inf, np.inf)

# Вычисление медианы
median_value = quad(integrand_mean, -np.inf, np.inf)[0]

print(f"Математическое ожидание E(X): {mean_value:.3f}")
print(f"Дисперсия Var(X): {variance_value:.3f}")
print(f"Медиана Me: {median_value:.3f}")

# Вычисление вероятности
probability = F(1.5) - F(1)

print(f"P{{X in [1, 1.5]}}: {probability}")
