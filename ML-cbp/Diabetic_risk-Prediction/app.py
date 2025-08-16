from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__,template_folder="main")

# Load model info at app startup
model_info = joblib.load('diabetes_model.pkl')
model = model_info['model']
feature_names = model_info['feature_names']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        input_data = {
            'Pregnancies': float(request.form['pregnancies']),
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['bloodpressure']),
            'SkinThickness': float(request.form['skinthickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetespedigree']),
            'Age': float(request.form['age']),
        }

        # Create dataframe with single row for prediction
        input_df = pd.DataFrame([input_data])

        # Make sure columns are in the right order as model expects
        input_df = input_df[feature_names]

        # Predict probability and class
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = int(prob > 0.5)

        # Determine risk level
        if prob > 0.75:
            risk_level = 'High'
        elif prob > 0.4:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'

        result = {
            'prediction': prediction,
            'probability': prob,
            'risk_level': risk_level
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
