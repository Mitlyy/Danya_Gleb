import matplotlib.pyplot as plt
import numpy as np
# plt.style.use(['science', 'notebook', 'grid'])

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_data_from_files(folder_path):
    # Получаем список файлов в папке
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # Проходимся по каждому файлу и строим график
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Читаем данные из файла
        data = np.loadtxt(file_path)
        
        # Создаем график
        plt.plot(data[:, 0], data[:, 1], label=file_name)
        
        # Добавляем название графика и легенду
        plt.title(file_name[:-4])
        plt.xlabel('Длина волны, нм')
        plt.ylabel('Интенсивность, (у.ед.)')
        plt.grid()
        plt.savefig(r'C:\Users\MSI User\Desktop\Синхронное детектирование\2\ '+ file_name[:-4])
        plt.cla()

        # # Отображаем график
        # plt.show()

# Замените 'путь/к/вашей/папке' на реальный путь к вашей папке с файлами txt
folder_path = r"E:\Code\Projects Python\ДАня мельник проект\Графички\Data\Измерения CdSe\21.12.jasco"
plot_data_from_files(folder_path)
