import numpy as np


try:
    data = np.loadtxt('iris.txt', delimiter=',')
except FileNotFoundError:
    print("Файл iris.txt не найден! Использую заглушку.")
    np.random.seed(42)
    d1 = np.column_stack([np.random.normal(5, 0.3, (50,4)), np.zeros(50)])
    d2 = np.column_stack([np.random.normal(6, 0.4, (50,4)), np.ones(50)])
    d3 = np.column_stack([np.random.normal(7, 0.5, (50,4)), np.full(50, 2)])
    data = np.vstack([d1, d2, d3])

X = data[:, :-1]
Y = data[:, -1].astype(int)

indices = np.arange(len(X))
np.random.shuffle(indices)
train_size = 100 # Как в примере методички
test_size = len(X) - train_size

train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train_raw = X[train_idx]
Y_train = Y[train_idx]
X_test_raw = X[test_idx]
Y_test = Y[test_idx]


num_bins = 10
X_train = np.zeros_like(X_train_raw, dtype=int)
X_test = np.zeros_like(X_test_raw, dtype=int)

bins_edges = []
for i in range(X_train_raw.shape[1]):
    min_val = X_train_raw[:, i].min()
    max_val = X_train_raw[:, i].max()

    edges = np.linspace(min_val, max_val, num_bins + 1)
    bins_edges.append(edges)
    

    X_train[:, i] = np.digitize(X_train_raw[:, i], edges) - 1
    X_test[:, i] = np.digitize(X_test_raw[:, i], edges) - 1
    

    X_train[:, i] = np.clip(X_train[:, i], 0, num_bins - 1)
    X_test[:, i] = np.clip(X_test[:, i], 0, num_bins - 1)

n_cls = 3 # 3 вида ирисов
m = 4     # 4 признака
n_total = len(Y_train)


Nj = np.zeros(n_cls)
for j in range(n_cls):
    Nj[j] = np.sum(Y_train == j)

Z = [list(range(num_bins)) for _ in range(m)]

M = []
for i in range(m):
    rw = []
    for j in range(n_cls):
        cl = []
        for s in range(num_bins):
            count = np.sum((Y_train == j) & (X_train[:, i] == s))
            cl.append(count)
        rw.append(cl)
    M.append(rw)

errors = 0
predictions = []

for k in range(len(X_test)):
    x0 = X_test[k]
    PP = np.zeros(n_cls)
    
    for j in range(n_cls):
        P = 1.0
        for i in range(m):
            s_val = x0[i]
            Ns_ij = M[i][j][s_val]

            if Nj[j] > 0:
                prob_cond = Ns_ij / Nj[j]
            else:
                prob_cond = 0
            
            P *= prob_cond
            
        PP[j] = P * (Nj[j] / n_total)
        
    pred_class = np.argmax(PP)
    predictions.append(pred_class)
    
    if pred_class != Y_test[k]:
        errors += 1

accuracy_error = errors / len(Y_test) # Доля ошибок
accuracy_score = 1 - accuracy_error

print(f"\nРезультаты Наивного Байеса:")
print(f"Всего тестовых: {len(Y_test)}")
print(f"Ошибок: {errors}")
print(f"Доля ошибок (Accuracy по формуле методички): {accuracy_error:.4f}")
print(f"Точность (Score): {accuracy_score:.4f}")

print("\nСравнение: Обычно KNN на этих данных дает ошибку около 0.04-0.07.")
if accuracy_error < 0.07:
    print("Байес показал результат сравнимый или лучше KNN!")
else:
    print("KNN, скорее всего, справился лучше, так как Байес чувствителен к дискретизации.")
