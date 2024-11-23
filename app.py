from flask import Flask, request, render_template
import joblib
import numpy as np

# Flask ilovasini yaratish
app = Flask(__name__)

# Saqlangan modelni yuklash
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Foydalanuvchi kiritgan ma'lumotlarni olish
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Bashorat uchun ma'lumotlarni tayyorlash
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(data)[0]

        # Natijani ko'rsatish
        result = "Yurak kasalligi bor" if prediction == 1 else "Yurak kasalligi yo'q"
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
