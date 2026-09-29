import numpy as np
import matplotlib.pyplot as plt

# Возраст, Доход, Класс
data_train = np.array([
    [18, 25, 0], [22, 100, 0], [30, 70, 0], [32, 120, 0],
    [24, 15, 1], [25, 22, 0], [32, 50, 1], [19, 45, 1],
    [22, 75, 0], [40, 90, 1]
])

X_train = data_train[:, :2]
Y_train = data_train[:, 2]

# Новые клиенты для классификации
X_new = np.array([
    [19, 27],  # Код 11
    [33, 54],  # Код 12
    [39, 88],  # Код 13
    [27, 30]   # Код 14
])

print("--- Линейная регрессия ---")

# Добавляем столбец единиц для свободного члена (bias)
X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
X_new_b = np.c_[np.ones(X_new.shape[0]), X_new]


XTX = np.dot(X_train_b.T, X_train_b)
XTY = np.dot(X_train_b.T, Y_train)

# Решаем систему
a = np.linalg.inv(XTX).dot(XTY)
print(f"Коэффициенты модели: {a}")

# Предсказание для новых клиентов
Y_pred_lin = np.dot(X_new_b, a)
# Округляем до ближайшего целого (0 или 1)
Y_class_lin = np.round(Y_pred_lin).astype(int)

for i, cls in enumerate(Y_class_lin):
    print(f"Клиент {11+i} (Age={X_new[i][0]}, Inc={X_new[i][1]}): Класс = {cls} (Score={Y_pred_lin[i]:.3f})")

print("\n--- Метод K ближайших соседей ---")

def knn_predict(X_train, Y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        # Вычисляем расстояния до всех точек обучающей выборки
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        # Находим индексы k ближайших соседей
        k_indices = np.argsort(distances)[:k]
        # Берем их классы
        k_labels = Y_train[k_indices]
        # Голосование: какой класс встречается чаще
        unique, counts = np.unique(k_labels, return_counts=True)
        pred_class = unique[np.argmax(counts)]
        predictions.append(pred_class)
    return np.array(predictions)

# Выбираем k (обычно нечетное, чтобы избежать ничьих)
k_val = 3 
Y_class_knn = knn_predict(X_train, Y_train, X_new, k=k_val)

for i, cls in enumerate(Y_class_knn):
    print(f"Клиент {11+i}: Класс = {cls} (при k={k_val})")

plt.figure(figsize=(10, 6))

plt.scatter(X_train[Y_train==0, 0], X_train[Y_train==0, 1], c='blue', marker='o', label='Class 0 (Train)', s=100)
plt.scatter(X_train[Y_train==1, 0], X_train[Y_train==1, 1], c='red', marker='^', label='Class 1 (Train)', s=100)

plt.scatter(X_new[:, 0], X_new[:, 1], c='green', marker='*', label='New Clients', s=200, edgecolors='black')

for i in range(len(X_new)):
    plt.text(X_new[i, 0]+0.5, X_new[i, 1]+2, f'#{11+i}', fontsize=12)

plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Классификация клиентов')
plt.legend()
plt.grid(True)
plt.show()