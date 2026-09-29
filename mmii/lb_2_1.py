import numpy as np
import matplotlib.pyplot as plt


try:
    data = np.loadtxt('vlg_temps.txt', delimiter=';')
except FileNotFoundError:
    print("Ошибка: Файл 'vlg_temps.txt' не найден!")
    exit()

years_all = data[:, 0]
months_all = data[:, 1]
temps_all = data[:, 2]

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

for month in range(1, 13):
    print(f"\n=== Анализ для месяца {month} ===")
    
    mask = (months_all == month)
    years = years_all[mask]
    temps = temps_all[mask].copy()
    
    for i in range(len(temps)):
        if temps[i] == 999.9:
            if 0 < i < len(temps) - 1:

                prev_val = temps[i-1]
                next_val = temps[i+1]
                if prev_val != 999.9 and next_val != 999.9:
                    temps[i] = (prev_val + next_val) / 2
                elif prev_val != 999.9:
                    temps[i] = prev_val
                elif next_val != 999.9:
                    temps[i] = next_val
                else:
                    temps[i] = 0
            elif i == 0 and len(temps) > 1 and temps[1] != 999.9:
                temps[i] = temps[1]
            elif i == len(temps) - 1 and temps[i-1] != 999.9:
                temps[i] = temps[i-1]
    
    valid_mask = temps != 999.9
    years = years[valid_mask]
    temps = temps[valid_mask]
    
    if len(years) < 5:
        print("Недостаточно данных для построения модели.")
        continue

    # --- 4. РАЗБИЕНИЕ НА ВЫБОРКИ (75% / 25%) ---
    N = len(years)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    split_idx = int(N * 0.75)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = years[train_idx]
    Y_train = temps[train_idx]
    X_test = years[test_idx]
    Y_test = temps[test_idx]
    
    # Нормализация года для экспоненты (чтобы e^2000 не было бесконечностью)
    # Сдвигаем начало отсчета на 1900 или 2000 и делим на 10
    X_train_norm = (X_train - 1900) / 10.0
    X_test_norm = (X_test - 1900) / 10.0
    
    results = {}
    
    # МОДЕЛЬ 1: Линейная y = ax + b ===
    A1 = np.array([[len(X_train), np.sum(X_train)],
                   [np.sum(X_train), np.dot(X_train, X_train)]])
    b1 = np.array([np.sum(Y_train), np.dot(X_train, Y_train)])
    try:
        coeffs1 = np.linalg.inv(A1).dot(b1)
        Y_pred1 = coeffs1[0] + coeffs1[1] * X_test
        mse1 = MSE(Y_test, Y_pred1)
        results['Linear'] = {'mse': mse1, 'coeffs': coeffs1, 'type': 1}
    except:
        pass

    # МОДЕЛЬ 2: Квадратичная y = ax^2 + bx + c ===
    Sx4 = np.sum(X_train**4)
    Sx3 = np.sum(X_train**3)
    Sx2 = np.sum(X_train**2)
    Sx1 = np.sum(X_train)
    Sn = len(X_train)
    
    Syx2 = np.sum(Y_train * X_train**2)
    Syx1 = np.sum(Y_train * X_train)
    Sy1 = np.sum(Y_train)
    
    A2 = np.array([[Sx4, Sx3, Sx2],
                   [Sx3, Sx2, Sx1],
                   [Sx2, Sx1, Sn]])
    b2 = np.array([Syx2, Syx1, Sy1])
    
    try:
        coeffs2 = np.linalg.inv(A2).dot(b2)
        Y_pred2 = coeffs2[0] * X_test**2 + coeffs2[1] * X_test + coeffs2[2]
        mse2 = MSE(Y_test, Y_pred2)
        results['Quadratic'] = {'mse': mse2, 'coeffs': coeffs2, 'type': 2}
    except:
        pass

    # МОДЕЛЬ 3: Экспоненциальная y = a*e^x + b*e^-x + c ===
    f0 = np.exp(X_train_norm)
    f1 = np.exp(-X_train_norm)
    f2 = np.ones_like(X_train_norm)
    
    A3 = np.array([
        [np.sum(f0*f0), np.sum(f0*f1), np.sum(f0*f2)],
        [np.sum(f1*f0), np.sum(f1*f1), np.sum(f1*f2)],
        [np.sum(f2*f0), np.sum(f2*f1), np.sum(f2*f2)]
    ])
    
    b3 = np.array([
        np.sum(Y_train * f0),
        np.sum(Y_train * f1),
        np.sum(Y_train * f2)
    ])
    
    try:
        coeffs3 = np.linalg.inv(A3).dot(b3)
        Y_pred3 = coeffs3[0] * np.exp((X_test - 1900)/10.0) + \
                  coeffs3[1] * np.exp(-(X_test - 1900)/10.0) + \
                  coeffs3[2]
        mse3 = MSE(Y_test, Y_pred3)
        results['Exponential'] = {'mse': mse3, 'coeffs': coeffs3, 'type': 3}
    except:
        pass
        
    # --- 5. ВЫВОД ЛУЧШЕЙ МОДЕЛИ ---
    if results:
        best_model_name = min(results, key=lambda k: results[k]['mse'])
        print(f"Лучшая модель: {best_model_name} (MSE={results[best_model_name]['mse']:.4f})")
        
        # plt.figure(figsize=(10, 5))
        # plt.scatter(X_train, Y_train, color='red', marker='x', label='Train')
        # plt.scatter(X_test, Y_test, color='green', marker='o', label='Test')
        
        # X_line = np.linspace(min(years), max(years), 100)
        
        # if best_model_name == 'Linear':
        #     c = results['Linear']['coeffs']
        #     Y_line = c[0] + c[1] * X_line
        # elif best_model_name == 'Quadratic':
        #     c = results['Quadratic']['coeffs']
        #     Y_line = c[0] * X_line**2 + c[1] * X_line + c[2]
        # else:
        #     c = results['Exponential']['coeffs']
        #     X_line_norm = (X_line - 1900) / 10.0
        #     Y_line = c[0] * np.exp(X_line_norm) + c[1] * np.exp(-X_line_norm) + c[2]
            
        # plt.plot(X_line, Y_line, label=f'Best Fit ({best_model_name})')
        # plt.title(f'Month {month} Temperature Regression')
        # plt.xlabel('Year')
        # plt.ylabel('Temperature')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    else:
        print("Не удалось построить ни одну модель.")

print("\nАнализ завершен.")