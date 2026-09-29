import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

try:
    data = np.loadtxt('vlg_temps.txt', delimiter=';')
except FileNotFoundError:
    print("Ошибка: Файл 'vlg_temps.txt' не найден!")
    exit()

years_all = data[:, 0]
months_all = data[:, 1]
temps_all = data[:, 2]

target_month = 1 

print(f"Анализ данных для месяца: {target_month}")

mask = (months_all == target_month)
years = years_all[mask]
temps = temps_all[mask].copy()

valid_mask = temps != 999.9
years = years[valid_mask]
temps = temps[valid_mask]

if len(years) < 3:
    print("Недостаточно данных для построения модели.")
    exit()

print(f"Найдено {len(years)} записей.")

X = years.reshape(-1, 1)
Y = temps

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

reg = linear_model.LinearRegression()
reg.fit(X_poly, Y)

intercept = reg.intercept_
coef = reg.coef_

print(f"Уравнение регрессии: y = {coef[0]:.4f} * x^2 + {coef[1]:.4f} * x + {intercept:.4f}")
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='green', label='Real Data')

X_line = np.linspace(min(years), max(years), 100).reshape(-1, 1)
X_line_poly = poly_features.transform(X_line)
Y_line = reg.predict(X_line_poly)

plt.plot(X_line, Y_line, color='red', linewidth=2, label='Quadratic Fit (Sklearn)')
plt.title(f"Temperature Trend for Month {target_month} (Volgograd)")
plt.xlabel("Year")
plt.ylabel("Average Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()