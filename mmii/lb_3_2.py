import numpy as np

try:
    data = np.loadtxt('iris.txt', delimiter=',')
except FileNotFoundError:
    print("Файл iris.txt не найден! Использую тестовые данные.")
    np.random.seed(42)
    data = np.vstack([
        np.column_stack([np.random.normal(5, 0.3, (50,4)), np.zeros(50)]),
        np.column_stack([np.random.normal(6, 0.4, (50,4)), np.ones(50)]),
        np.column_stack([np.random.normal(7, 0.5, (50,4)), np.full(50, 2)])
    ])

X = data[:, :-1]
Y = data[:, -1]

indices = np.arange(len(X))
np.random.shuffle(indices)

train_size = int(0.7 * len(X))
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

print(f"Размер обучения: {len(X_train)}, Размер теста: {len(X_test)}")

def knn_classifier(X_tr, Y_tr, X_te, k):
    preds = []
    for x in X_te:
        dists = np.sqrt(np.sum((X_tr - x)**2, axis=1))
        nearest = np.argsort(dists)[:k]
        labels = Y_tr[nearest]
        # Голосование
        vals, cnts = np.unique(labels, return_counts=True)
        preds.append(vals[np.argmax(cnts)])
    return np.array(preds)

k = 5 # Можно менять
Y_pred = knn_classifier(X_train, Y_train, X_test, k=k)

# Количество неверных предсказаний
errors = np.sum(Y_pred != Y_test)
total_test = len(Y_test)

error_rate = errors / total_test
accuracy = 1 - error_rate

print(f"\nРезультаты при k={k}:")
print(f"Всего тестовых объектов: {total_test}")
print(f"Количество ошибок: {errors}")
print(f"Доля ошибок (по формуле из задания): {error_rate:.4f}")
print(f"Точность (Accuracy): {accuracy:.4f}")

# Проверим для разных k
print("\nЗависимость ошибки от k:")
for k_val in [1, 3, 5, 7, 9]:
    yp = knn_classifier(X_train, Y_train, X_test, k=k_val)
    err = np.sum(yp != Y_test) / len(Y_test)
    print(f"k={k_val}: Error={err:.4f}, Accuracy={1-err:.4f}")