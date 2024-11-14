# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

# Define the prediction route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([int_features])  # Scale input features and convert to 2D array

    # Make prediction
    prediction = model.predict(final_features)
    weather_conditions = {0: "Clear", 1: "Cloudy", 2: "Rain", 3: "Snow"}
    output = weather_conditions[int(prediction[0])]

    # Return result
    return render_template('index.html', prediction_text='Weather Condition: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
