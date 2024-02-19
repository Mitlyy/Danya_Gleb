import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def plot_3d_function():
    # Функция, которую мы будем рассматривать
    def func(x, y, z):
        # print(z)
        return np.sin(x) * np.cos(y) * z

    # Создаем фигуру и оси
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Задаем начальные значения x, y и z
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    z = np.linspace(-5, 5, 100)
    
    X, Y, Z = np.meshgrid(x,y,z, indexing= "xy")

    # Создаем сетку для x и y
    # X, Y = np.meshgrid(x, y)

    # Инициализируем значение z0
    z0 = 0

    # Создаем график функции для z0
    Z0 = func(X, Y, Z)
    surf = ax.plot_surface(X[:,:,0], Y[:,:,0], Z0[:,:, 0], cmap='viridis')

    # Добавляем ползунок для z
    ax_z = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_z = Slider(ax_z, 'z', 0, 100, valinit=z0, valstep = 1)

    # Функция обновления графика при изменении ползунка
    def update(val):
        nonlocal surf
        surf.remove()  # Удаляем предыдущий график
        surf = ax.plot_surface(X[:,:,0], Y[:,:,0], Z0[:,:, slider_z.val], cmap='viridis')  # Строим новый график
        fig.canvas.draw_idle()  # Перерисовываем фигуру

    # Связываем обновление графика с изменением значения ползунка
    slider_z.on_changed(update)

    # Устанавливаем метки осей и заголовок
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Function Plot')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    # Показываем график
    plt.show()

# Теперь вызовите функцию plot_3d_function(), чтобы увидеть график
plot_3d_function()
