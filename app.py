from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import os

# Load the trained model
model = pickle.load(open('tree_model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Make sure this exists in 'templates/'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # If the request is JSON (API call)
        if request.is_json:
            data = request.get_json()
            N = float(data['N'])
            P = float(data['P'])
            K = float(data['K'])
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            ph = float(data['ph'])
            rainfall = float(data['rainfall'])
        else:
            # Request is from HTML form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

        # Convert input to DataFrame
        user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Make prediction
        prediction = model.predict(user_input)[0]

        if request.is_json:
            return jsonify({"prediction": prediction})
        else:
            return render_template('result.html', prediction=prediction)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        else:
            return f"Error occurred: {e}"

# Production-compatible host and port for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
