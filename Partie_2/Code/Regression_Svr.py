# -------------------------------------------------------------
# SVR et Régression Linéaire - Analyse complète Heating Load
# -------------------------------------------------------------

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Chargement des données
df = pd.read_csv("cleaned_building_data.csv")

# Variables explicatives et cible
X = df.drop(['Heating_Load', 'Cooling_Load'], axis=1)
y = df['Heating_Load']

# 3. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Données prêtes :")
print("Taille train :", X_train.shape)
print("Taille test  :", X_test.shape)
print("-" * 50)

# 4. Régression Linéaire
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Régression Linéaire : MSE={mse_lr:.4f}, MAE={mae_lr:.4f}, R²={r2_lr:.4f}")

# 5. SVR (Kernel RBF)
svr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=50, epsilon=0.1))
])
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print(f"SVR : MSE={mse_svr:.4f}, MAE={mae_svr:.4f}, R²={r2_svr:.4f}")
print("-" * 50)

# 6. Scatter plot Réel vs Prédit
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, label='Linéaire')
plt.scatter(y_test, y_pred_svr, alpha=0.7, label='SVR', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Réel vs Prédit : Linéaire vs SVR")
plt.legend()
plt.grid(True)
plt.show()

# 7. Courbes comparatives (100 premiers points)
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label='Réel', marker='o')
plt.plot(y_pred_lr[:100], label='Lin.', marker='x')
plt.plot(y_pred_svr[:100], label='SVR', marker='s')
plt.title("Comparaison sur 100 premiers points")
plt.legend()
plt.show()

# 8. Histogramme des résidus
residuals_lr = y_test - y_pred_lr
residuals_svr = y_test - y_pred_svr

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(residuals_lr, bins=30, kde=True, color='blue')
plt.title("Résidus - Linéaire")
plt.subplot(1,2,2)
sns.histplot(residuals_svr, bins=30, kde=True, color='orange')
plt.title("Résidus - SVR")
plt.show()

# 9. Boxplot des erreurs
error_df = pd.DataFrame({
    'Linéaire': residuals_lr,
    'SVR': residuals_svr
})
plt.figure(figsize=(6,5))
sns.boxplot(data=error_df)
plt.title("Boxplot des résidus")
plt.ylabel("Erreur")
plt.show()

# 10. Scatter densité SVR
plt.figure(figsize=(6,6))
sns.kdeplot(x=y_test, y=y_pred_svr, fill=True, cmap="Reds", thresh=0.05)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions SVR")
plt.title("Densité des prédictions SVR")
plt.show()

# 11. Heatmap des corrélations
plt.figure(figsize=(10,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# 12. Scatter plots de chaque variable vs Heating Load
features = X.columns
plt.figure(figsize=(15,10))
for i, col in enumerate(features, 1):
    plt.subplot(3,3,i)
    plt.scatter(df[col], y, alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("Heating Load")
plt.suptitle("Scatter plots des variables vs Heating Load", y=1.02)
plt.tight_layout()
plt.show()

# 13. Résidus vs Prédictions
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_pred_lr, residuals_lr, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Prédictions Linéaire")
plt.ylabel("Résidus")
plt.title("Résidus vs Prédictions - Linéaire")

plt.subplot(1,2,2)
plt.scatter(y_pred_svr, residuals_svr, alpha=0.7, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Prédictions SVR")
plt.ylabel("Résidus")
plt.title("Résidus vs Prédictions - SVR")
plt.tight_layout()
plt.show()

# 14. Erreurs cumulées absolues
cum_abs_error_lr = np.cumsum(np.abs(residuals_lr))
cum_abs_error_svr = np.cumsum(np.abs(residuals_svr))
plt.figure(figsize=(8,5))
plt.plot(cum_abs_error_lr, label="Linéaire", color='blue')
plt.plot(cum_abs_error_svr, label="SVR", color='orange')
plt.xlabel("Index des données test")
plt.ylabel("Erreur absolue cumulée")
plt.title("Comparaison des erreurs cumulées")
plt.legend()
plt.show()

# 15. KDE des valeurs réelles vs prédites
plt.figure(figsize=(8,5))
sns.kdeplot(y_test, label='Réel', fill=True)
sns.kdeplot(y_pred_lr, label='Linéaire', fill=True)
sns.kdeplot(y_pred_svr, label='SVR', fill=True)
plt.title("Distribution des valeurs réelles vs prédites")
plt.xlabel("Heating Load")
plt.ylabel("Densité")
plt.legend()
plt.show()

# 16. Résumé des performances
results = pd.DataFrame({
    'Modèle': ['Régression Linéaire', 'SVR (RBF)'],
    'MSE': [mse_lr, mse_svr],
    'MAE': [mae_lr, mae_svr],
    'R²': [r2_lr, r2_svr]
})
print("\nRésumé des performances :")
print(results)

plt.figure(figsize=(6,4))
plt.bar(results['Modèle'], results['R²'], color=['blue','orange'])
plt.title("Comparaison des performances (R²)")
plt.ylabel("R²")
plt.show()

# 17. Meilleur modèle
best_model = results.loc[results['R²'].idxmax(), 'Modèle']
print(f"\nLe meilleur modèle pour la prédiction du Heating Load est : {best_model}")
