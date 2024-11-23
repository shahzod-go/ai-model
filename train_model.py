import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ma'lumotlar to'plamini yuklash
data = pd.read_csv('C:/web_suniy/heart.csv')  # To'liq manzilni kiriting

# X va y qiymatlarini ajratish
X = data.drop(columns=['target'])
y = data['target']

# Ma'lumotlarni o'rgatish va sinov to'plamlariga bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini yaratish va o'rgatish
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Modelni saqlash
joblib.dump(model, 'heart_disease_model.pkl')
print("Model muvaffaqiyatli saqlandi!")
