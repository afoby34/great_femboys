import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

area = np.array([69.5, 37.6, 69.7, 39.9, 51.2, 24.9, 69.6, 68.8, 69.6, 64.5, 105, 34.2, 29.4, 31.8, 29.6, 60, 75.4, 76.4, 51.1, 50.3])
year = np.array([2008, 2011, 2013, 2023, 2005, 2020, 2016, 2019, 2013, 2004, 2002, 2023, 2025, 2005, 2014, 2019, 2025, 2010, 2013, 2016])
price = np.array([5.87, 3.94, 6.99, 6.6, 6.2, 3.699, 6.6, 8.2, 6.15, 8.1, 10.49, 4.92, 3.6, 3.95, 3.999, 8.65, 9.1, 6.6, 6.5, 6.2])

n = len(price)
X = np.column_stack((np.ones(n), area, year))
Y = price

# Формула a = (X^T * X)^-1 * X^T * Y
XTX = np.dot(X.T, X)
XTY = np.dot(X.T, Y)
coeffs = np.linalg.inv(XTX).dot(XTY)

print(f"Коэффициенты: a0={coeffs[0]}, a1(area)={coeffs[1]}, a2(year)={coeffs[2]}")

# 3D График
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(area, year, price, c='r', marker='o')

# Сетка для поверхности
area_grid, year_grid = np.meshgrid(np.linspace(min(area), max(area), 10),
                                   np.linspace(min(year), max(year), 10))
price_grid = coeffs[0] + coeffs[1]*area_grid + coeffs[2]*year_grid

ax.plot_surface(area_grid, year_grid, price_grid, alpha=0.5, color='blue')
ax.set_xlabel('Площадь')
ax.set_ylabel('Год')
ax.set_zlabel('Цена')
plt.show()