import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

scores_info = [100, 94, 93, 92, 89, 97, 96, 78, 78, 74, 68, 75, 62, 54, 75, 64, 59, 70, 56, 67, 59, 56, 59, 46, 46, 51, 48, 51, 46]

# Создаем Series
s = pd.Series(scores_info)

# 1. Статистические характеристики
mean_val = s.mean()
var_val = s.var()
std_val = s.std()
median_val = s.median()
mode_val = s.mode()[0]
range_val = s.max() - s.min()
coef_var = 100 * std_val / mean_val

print("--- Статистика (Informatics) ---")
print(f"Mean: {mean_val:.2f}")
print(f"Var (ddof=1): {var_val:.2f}")
print(f"Std: {std_val:.2f}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
print(f"Range: {range_val}")
print(f"Coeff of Variation: {coef_var:.2f}%")

# 2. Гистограмма и нормальное распределение
plt.figure(figsize=(10, 6))


s.plot(kind="hist", bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram (Density)')

xmin, xmax = plt.xlim()
x_norm = np.linspace(xmin, xmax, 100)
p_norm = st.norm.pdf(x_norm, mean_val, std_val)
plt.plot(x_norm, p_norm, 'r-', linewidth=2, label=f'Normal Dist (mu={mean_val:.1f}, sigma={std_val:.1f})')

plt.title("Distribution of Scores (Informatics)")
plt.xlabel("Score")
plt.ylabel("Density / Frequency")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
# Эмпирическая CDF
sorted_data = np.sort(s)
y_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
plt.plot(sorted_data, y_cdf, 'bo-', label='Empirical CDF')

# Теоретическая CDF
p_cdf = st.norm.cdf(sorted_data, mean_val, std_val)
plt.plot(sorted_data, p_cdf, 'r-', label='Theoretical Normal CDF')

plt.title("Cumulative Distribution Function")
plt.xlabel("Score")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()