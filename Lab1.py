import numpy as np
import matplotlib.pyplot as plt

# Определение функции правой части дифференциального уравнения
def f(x, y):
    return (1 + x) * np.exp(-y)

# Аналитическое решение
def analytical_solution(x):
    return np.log((x**2 + 2*x)/2) - (15 - 24)/25

# Начальное условие
x0 = 0.4
y0 = 1.0

# Отрезок интегрирования
a = 0.4
b = 1.9
n = 16  # Число шагов

# Метод Эйлера
x_euler = np.linspace(a, b, n)
y_euler = np.zeros_like(x_euler)
y_euler[0] = y0
h = (b - a) / (n - 1)
for i in range(1, n):
    y_euler[i] = y_euler[i-1] + h * f(x_euler[i-1], y_euler[i-1])

# Метод Эйлера-Коши
x_euler_cromer = np.linspace(a, b, n)
y_euler_cromer = np.zeros_like(x_euler_cromer)
y_euler_cromer[0] = y0
for i in range(1, n):
    k1 = h * f(x_euler_cromer[i-1], y_euler_cromer[i-1])
    k2 = h * f(x_euler_cromer[i-1] + h/2, y_euler_cromer[i-1] + k1/2)
    y_euler_cromer[i] = y_euler_cromer[i-1] + k2

# Аналитическое решение
x_analytical = np.linspace(a, b, 100)
y_analytical = analytical_solution(x_analytical)

# Вывод решений в таблицах
print("Решение методом Эйлера:")
print("x\ty")
for i in range(n):
    print(f"{x_euler[i]:.4f}\t{y_euler[i]:.4f}")

print("\nРешение методом Эйлера-Коши:")
print("x\ty")
for i in range(n):
    print(f"{x_euler_cromer[i]:.4f}\t{y_euler_cromer[i]:.4f}")

# Определение разницы между решениями
diff_euler = abs(y_euler - analytical_solution(x_euler))
diff_euler_cromer = abs(y_euler_cromer - analytical_solution(x_euler_cromer))

# Таблица разницы между решениями
print("\nТаблица разницы между решениями:")
print("x\tРазница (Эйлер)\tРазница (Эйлер-Коши)")
for i in range(n):
    print(f"{x_euler[i]:.4f}\t{diff_euler[i]:.6f}\t{diff_euler_cromer[i]:.6f}")

# Отображение результатов
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

ax1.plot(x_euler, y_euler, label='Метод Эйлера')
ax1.plot(x_analytical, y_analytical, label='Аналитическое решение', linestyle='dashed')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Решение дифференциального уравнения (Метод Эйлера)')
ax1.legend()
ax1.grid()

ax2.plot(x_euler_cromer, y_euler_cromer, label='Метод Эйлера-Коши')
ax2.plot(x_analytical, y_analytical, label='Аналитическое решение', linestyle='dashed')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Решение дифференциального уравнения (Метод Эйлера-Коши)')
ax2.legend()
ax2.grid()

ax3.plot(x_euler, diff_euler, label='Разница (Метод Эйлера)', color='blue')
ax3.plot(x_euler_cromer, diff_euler_cromer, label='Разница (Метод Эйлера-Коши)', color='red')
ax3.set_xlabel('x')
ax3.set_ylabel('Разница')
ax3.set_title('Разница между численными и аналитическим решениями')
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.show()