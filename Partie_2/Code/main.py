import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("cleaned_building_data.csv")


X = df.drop(['Heating_Load', 'Cooling_Load'], axis=1)
y_heating = df['Heating_Load']
y_cooling = df['Cooling_Load']


X_train, X_test, y_train, y_test = train_test_split(
    X, y_heating, test_size=0.2, random_state=42
)

print("Taille du jeu d'entraînement :", X_train.shape)
print("Taille du jeu de test :", X_test.shape)
