import numpy as np
import matplotlib.pyplot as plt


try:

    data = np.loadtxt('vlg_temps.txt', delimiter=';')
except FileNotFoundError:
    print("Ошибка: Файл 'vlg_temps.txt' не найден!")
    exit()

years_all = data[:, 0]
months_all = data[:, 1]
temps_all = data[:, 2]


unique_years = np.unique(years_all)
avg_yearly_temps = []
valid_years_list = []

print("Расчет среднегодовых температур...")
for year in unique_years:

    mask = (years_all == year)
    temps_year = temps_all[mask]
    
    temps_valid = temps_year[temps_year != 999.9]

    if len(temps_valid) > 6:
        avg_temp = np.mean(temps_valid)
        avg_yearly_temps.append(avg_temp)
        valid_years_list.append(year)

avg_yearly_temps = np.array(avg_yearly_temps)
valid_years = np.array(valid_years_list)

if len(avg_yearly_temps) == 0:
    print("Не удалось рассчитать ни одной среднегодовой температуры.")
    exit()

print(f"Успешно рассчитано для {len(valid_years)} лет.")

plt.figure(figsize=(12, 6))
plt.plot(valid_years, avg_yearly_temps, 'b-o', markersize=4, label='Avg Temp')
plt.title('Average Annual Temperature in Volgograd (Real Data)')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.show()
mean_val = np.mean(avg_yearly_temps)
std_val = np.std(avg_yearly_temps)
var_val = np.var(avg_yearly_temps)

print(f"\nСтатистика среднегодовых температур:")
print(f"Среднее: {mean_val:.2f}")
print(f"СКО: {std_val:.2f}")
print(f"Дисперсия: {var_val:.2f}")

# Находим год с максимальным отклонением
deviations = np.abs(avg_yearly_temps - mean_val)
max_dev_idx = np.argmax(deviations)
min_dev_idx = np.argmin(deviations) # Для интереса: минимальное отклонение

print(f"\nГод с МАКС. отклонением от средней: {int(valid_years[max_dev_idx])} "
      f"(Temp: {avg_yearly_temps[max_dev_idx]:.2f}, Dev: {deviations[max_dev_idx]:.2f})")
      
print(f"Год с МИН. отклонением от средней: {int(valid_years[min_dev_idx])} "
      f"(Temp: {avg_yearly_temps[min_dev_idx]:.2f}, Dev: {deviations[min_dev_idx]:.2f})")

# Разбиваем диапазон на 10 интервалов
t_min = np.min(avg_yearly_temps)
t_max = np.max(avg_yearly_temps)

bins_edges = np.linspace(t_min, t_max, 11) # 11 границ дают 10 интервалов
counts, bin_edges = np.histogram(avg_yearly_temps, bins=bins_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

print(f"\nЧастоты по интервалам: {counts}")

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Создаем гладкую линию для нормального распределения
x_smooth = np.linspace(t_min - 1, t_max + 1, 100)
y_smooth = normal_pdf(x_smooth, mean_val, std_val)

# Масштабируем PDF под гистограмму (умножаем на кол-во точек * ширину бина)
bin_width = bin_edges[1] - bin_edges[0]
y_smooth_scaled = y_smooth * len(avg_yearly_temps) * bin_width

plt.figure(figsize=(10, 6))
# Полигон частот
plt.plot(bin_centers, counts, 'ro-', label='Frequency Polygon')
# График нормального распределения
plt.plot(x_smooth, y_smooth_scaled, 'b-', linewidth=2, label=f'Normal Dist (mu={mean_val:.2f}, sigma={std_val:.2f})')

plt.title('Distribution of Average Annual Temperatures vs Normal Law')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

print("\nГотово! Все графики построены на реальных данных.")