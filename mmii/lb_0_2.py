import numpy as np

A = np.loadtxt("matrix.csv", delimiter=";")
print("Исходная матрица A:\n", A)

AtA = A.T @ A
print("\nA^T @ A:\n", AtA)

if np.linalg.matrix_rank(AtA) == AtA.shape[0]:
    result = np.linalg.inv(AtA) @ A.T
    print("\n(A^T @ A)^-1 @ A^T:\n", result)
else:
    print("Матрица AtA вырождена. Используем псевдообразную:")
    print(np.linalg.pinv(AtA) @ A.T)