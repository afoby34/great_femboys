import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

X = np.array([22, 68, 108, 137, 255, 315, 390, 405, 685, 700, 1100])
Y = np.array([7.5, 24, 32, 47, 93, 120, 134, 144, 245, 260, 380])

X_2d = X.reshape(-1, 1)

reg = linear_model.LinearRegression()
reg.fit(X_2d, Y)

a = reg.intercept_ # свободный член
b = reg.coef_[0]   # коэффициент при x

print(f"Уравнение регрессии: y = {a:.2f} + {b:.2f} * x")

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='red', label='Data')
plt.plot(X, reg.predict(X_2d), color='blue', linewidth=2, label=f'Regression Line (k={b:.2f})')
plt.title("Hubble's Law Linear Regression")
plt.xlabel("Distance (mln light years)")
plt.ylabel("Speed (hundreds miles/sec)")
plt.legend()
plt.grid(True)
plt.show()