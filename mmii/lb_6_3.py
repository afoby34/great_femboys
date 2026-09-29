import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

try:
    data = np.loadtxt('iris.txt', delimiter=',')
except FileNotFoundError:
    print("Файл не найден, использую тестовые данные")
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = np.column_stack((iris.data, iris.target))

X = data[:, :-1]
Y = data[:, -1]

# Разбиение на выборки (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Список классификаторов
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=10)
}

print(f"{'Method':<20} | {'Accuracy':<10} | {'MAE':<10} | {'MSE':<10}")
print("-" * 60)

for name, clf in classifiers.items():
    # Обучение
    clf.fit(X_train, y_train)
    # Предсказание
    y_pred = clf.predict(X_test)
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"{name:<20} | {acc:.4f}     | {mae:.4f}     | {mse:.4f}")