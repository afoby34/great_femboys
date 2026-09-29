import numpy as np
import matplotlib.pyplot as plt

# Прикладная информатика (29 значений)
info = np.array([100, 94, 93, 92, 89, 97, 96, 78, 78, 74, 68, 75, 62, 54, 75, 64, 59, 70, 56, 67, 59, 56, 59, 46, 46, 51, 48, 51, 46])
# Биология (26 значений)
bio = np.array([91, 84, 73, 59, 72, 77, 55, 77, 64, 56, 74, 64, 64, 55, 71, 61, 46, 79, 57, 43, 52, 40, 65, 57, 58, 57])
# Статистика (22 значения)
stat = np.array([90, 73, 64, 83, 65, 67, 48, 54, 45, 54, 51, 75, 46, 46, 46, 46, 42, 40, 41, 42, 42, 40])

datasets = {
    "Informatics": info,
    "Biology": bio,
    "Statistics": stat
}

def analyze_and_plot(name, data):
    print(f"\n=== Анализ направления: {name} ===")
    n = len(data)
    
    # 1. Статистический ряд (биннинг)
    # Разбиваем [0, 100] на 10 интервалов: 0-10, 10-20, ..., 90-100
    bins_edges = np.arange(0, 101, 10) # границы бинов
    # hist возвращает частоты и границы
    counts, bin_edges = np.histogram(data, bins=bins_edges)
    
    # Середины интервалов для построения полигона
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print(f"Частоты по интервалам: {counts}")
    
    # 2. Вычисление характеристик
    mean_val = np.mean(data)
    var_val = np.var(data) # Выборочная дисперсия (нормированная на N)
    std_val = np.std(data) # СКО
    median_val = np.median(data)
    range_val = np.max(data) - np.min(data)
    coef_var = (std_val / mean_val) * 100 if mean_val != 0 else 0
    
    # Мода (самое частое значение в исходных данных)
    unique, counts_raw = np.unique(data, return_counts=True)
    mode_val = unique[np.argmax(counts_raw)]
    
    print(f"Среднее: {mean_val:.2f}")
    print(f"Дисперсия: {var_val:.2f}")
    print(f"СКО: {std_val:.2f}")
    print(f"Медиана: {median_val:.2f}")
    print(f"Мода: {mode_val}")
    print(f"Размах: {range_val}")
    print(f"Коэф. вариации: {coef_var:.2f}%")
    
    # 3. Графики
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Гистограмма частот
    axes[0].bar(bin_centers, counts, width=10, align='center', edgecolor='black', color='skyblue')
    axes[0].set_title(f'Histogram of Frequencies ({name})')
    axes[0].set_xlabel('Score Interval')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xticks(np.arange(5, 100, 10))
    
    # Полигон частот
    axes[1].plot(bin_centers, counts, 'ro-', label='Polygon')
    axes[1].set_title(f'Frequency Polygon ({name})')
    axes[1].set_xlabel('Score Interval Center')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)
    

    cum_counts = np.cumsum(counts)

    cum_freq = cum_counts / n
    
    plot_x = np.concatenate(([bin_edges[0]], bin_edges[1:]))
    plot_y = np.concatenate(([0], cum_freq))
    
    axes[2].step(plot_x, plot_y, where='post', label='CDF', color='green')
    axes[2].set_title(f'Distribution Function ({name})')
    axes[2].set_xlabel('Score')
    axes[2].set_ylabel('F(x)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

for name, data in datasets.items():
    analyze_and_plot(name, data)

