import numpy as np

data_train = np.array([
    [18, 25, 0], [22, 100, 0], [30, 70, 0], [32, 120, 0],
    [24, 15, 1], [25, 22, 0], [32, 50, 1], [19, 45, 1],
    [22, 75, 0], [40, 90, 1]
])

X_train = data_train[:, :2].astype(int)
Y_train = data_train[:, 2].astype(int)

# Новые клиенты
X_new = np.array([
    [19, 27],  # Код 11
    [33, 54],  # Код 12
    [39, 88],  # Код 13
    [27, 30]   # Код 14
]).astype(int)

n_cls = 2 
m = 2     

all_data = np.vstack([X_train, X_new])
min_vals = all_data.min(axis=0)
max_vals = all_data.max(axis=0)

Z = []
for i in range(m):
    Z.append(list(range(min_vals[i], max_vals[i] + 1)))

print("Диапазоны признаков:", Z)

Nj = np.zeros(n_cls)
for j in range(n_cls):
    Nj[j] = np.sum(Y_train == j)

print("Количество объектов в классах (Nj):", Nj)
n_total = len(Y_train)

M = []
for i in range(m):
    rw = [] 
    for j in range(n_cls):
        cl = [] 
        for s_val in Z[i]:
            count = np.sum((Y_train == j) & (X_train[:, i] == s_val))
            cl.append(count)
        rw.append(cl)
    M.append(rw)

print("\nРезультаты классификации (со сглаживанием):")
for idx, x0 in enumerate(X_new):
    PP = np.zeros(n_cls) 
    
    for j in range(n_cls):
        P = 1.0
        for i in range(m):
            val = x0[i]
            if val in Z[i]:
                s_index = Z[i].index(val)
                Ns_ij = M[i][j][s_index]
                
                k_i = len(Z[i])
                
                # Применяем сглаживание Лапласа
                prob_cond = (Ns_ij + 1) / (Nj[j] + k_i)
                P *= prob_cond
            else:

                P *= (1 / (Nj[j] + len(Z[i]))) 
        
        PP[j] = P * (Nj[j] / n_total)
    
    # Нормализация
    sum_PP = np.sum(PP)
    if sum_PP > 0:
        PP = PP / sum_PP
    
    predicted_class = np.argmax(PP)
    print(f"Клиент {11+idx} (Age={x0[0]}, Inc={x0[1]}): Класс={predicted_class}, Вероятности={PP}")