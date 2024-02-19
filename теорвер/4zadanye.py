# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Загрузка данных из файла exams_dataset.csv
# df = pd.read_csv('exams_dataset.csv')

# # Рассчитываем выборочное среднее
# mean = df['math score'].mean()

# # Рассчитываем выборочную дисперсию
# variance = df['math score'].var()

# # Рассчитываем выборочное стандартное отклонение
# std_dev = df['math score'].std()

# # Рассчитываем выборочную медиану
# median = df['math score'].median()

# # Рассчитываем верхний и нижний квартили
# q1 = df['math score'].quantile(0.25)
# q3 = df['math score'].quantile(0.75)

# # Рассчитываем межквартильный размах
# iqr = q3 - q1

# # Рассчитываем 95% асимптотические доверительные интервалы для математического и дисперсии для результата по математике math score
# n = len(df['math score'])
# z = 1.96
# se = std_dev / np.sqrt(n)
# lcb = mean - z * se
# ucb = mean + z * se

# # Выводим результаты
# print('Выборочное среднее: {:.2f}'.format(mean))
# print('Выборочная дисперсия: {:.2f}'.format(variance))
# print('Выборочное стандартное отклонение: {:.2f}'.format(std_dev))
# print('Выборочная медиана: {:.2f}'.format(median))
# print('Верхний квартиль: {:.2f}'.format(q3))
# print('Нижний квартиль: {:.2f}'.format(q1))
# print('Межквартильный размах: {:.2f}'.format(iqr))
# print('95% асимптотические доверительные интервалы для математического и дисперсии для результата по математике math score: ({:.2f}, {:.2f})'.format(lcb, ucb))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t

# Загрузка данных
df = pd.read_csv("exams_dataset.csv")

# Функция для расчета 95% доверительного интервала
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(n)
    t_value = t.ppf(0.05 / 2, n - 1)
    margin_of_error = t_value * std_error
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

# Рассчитываем показатели для всего набора данных по math score
math_scores = df['math score']
mean_math = np.mean(math_scores)
var_math = np.var(math_scores, ddof=1)
std_math = np.std(math_scores, ddof=1)
median_math = np.median(math_scores)
q1_math = np.percentile(math_scores, 25)
q3_math = np.percentile(math_scores, 75)
iqr_math = q3_math - q1_math
ci_math_mean = calculate_confidence_interval(math_scores)[::-1]  # инвертируем для верного отображения на графике
ci_math_var = calculate_confidence_interval(math_scores, np.var)[::-1]

# Рассчитываем показатели для каждой комбинации пола и посещения подготовительных курсов
for gender in df['gender'].unique():
    for test_prep in df['test preparation course'].unique():
        subset = df[(df['gender'] == gender) & (df['test preparation course'] == test_prep)]
        math_scores_subset = subset['math score']
        mean_subset = np.mean(math_scores_subset)
        var_subset = np.var(math_scores_subset, ddof=1)
        std_subset = np.std(math_scores_subset, ddof=1)
        median_subset = np.median(math_scores_subset)
        q1_subset = np.percentile(math_scores_subset, 25)
        q3_subset = np.percentile(math_scores_subset, 75)
        iqr_subset = q3_subset - q1_subset
        ci_subset_mean = calculate_confidence_interval(math_scores_subset)[::-1]
        ci_subset_var = calculate_confidence_interval(math_scores_subset, np.var)[::-1]

        # Гистограмма
        plt.figure(figsize=(12, 6))
        sns.histplot(math_scores_subset, kde=True, color='blue', bins=20)
        plt.title(f'Гистограмма Math Scores ({gender}, {test_prep})')
        plt.xlabel('Math Score')
        plt.ylabel('Frequency')
        plt.show()

        # Box-plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='math score', y='gender', hue='test preparation course', data=subset)
        plt.title(f'Box Plot Math Scores ({gender}, {test_prep})')
        plt.show()

# Рассчитываем выборочный коэффициент корреляции
correlation_coefficient = df['math score'].corr(df['test preparation course'].map({'none': 0, 'completed': 1}))

# Вывод результатов

print(f"Среднее: {mean_math:.3f}")
print(f"Дисперсия: {var_math:.3f}")
print(f"Стандартное отклонение: {std_math:.3f}")
print(f"Медиана: {median_math:.3f}")
print(f"Q1: {q1_math:.3f}")
print(f"Q3: {q3_math:.3f}")
print(f"IQR: {iqr_math:.3f}")
print(f"95% Среднее: {ci_math_mean}")
print(f"95% Дисперсия: {ci_math_var}")

print(f"\nВыборочный коэффициент корреляции: {correlation_coefficient:.3f}")
