import numpy as np
A = np.array([[2, 3, -1],
              [5, -2, 2], 
              [3, -1, 4]], dtype=float)
b = np.array([5, 13, 7], dtype=float)
x_inv = np.linalg.inv(A).dot(b)
x_solve = np.linalg.solve(A, b)
print("Метод обратной матрицы: ", x_inv)
print("Встроенная функция: ", x_solve)

det_A = np.linalg.det(A)
n = len(b)
x_cramer = np.zeros(n)
for i in range(n):
    A_i = A.copy()
    A_i[:,i] = b
    x_cramer[i] = np.linalg.det(A_i)/det_A
print("Метод Крамера: ", x_cramer) 