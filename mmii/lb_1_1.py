import numpy as np
import matplotlib.pyplot as plt

X = np.array([7.5, 24, 32, 47, 93, 120, 134, 144, 245, 260, 380])
Y = np.array([22, 68, 108, 137, 255, 315, 390, 405, 685, 700, 1100])

n = len(X)
A_mat = np.array([[n, np.sum(X)],
                  [np.sum(X), np.dot(X, X)]])
b_vec = np.array([np.sum(Y), np.dot(X, Y)])

a = np.linalg.inv(A_mat) @ b_vec
print(f"y = {a[0]:.2f} + {a[1]:.2f} * x")

plt.scatter(X, Y, color='red', label='Наблюдения')
X_line = np.linspace(min(X), max(X), 100)
plt.plot(X_line, a[0] + a[1]*X_line, 'b-', label='Линейная регрессия')
plt.xlabel('Расстояние (млн св.лет)')
plt.ylabel('Скорость (сотни миль/с)')
plt.legend()
plt.grid(True)
plt.show()