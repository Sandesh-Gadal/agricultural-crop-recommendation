from flask import Flask, request, render_template
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
    return render_template('index.html')  # Create this file in templates folder

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form values
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Convert into DataFrame for model
            user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                      columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

            # Make prediction
            prediction = model.predict(user_input)[0]

            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return f"Error occurred: {e}"

# if __name__ == '__main__':
#     app.run(debug=True)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sets PORT env var
    app.run(host='0.0.0.0', port=port, debug=False)
