from flask import Flask, request, jsonify, render_template

import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ridge

application = Flask(__name__)
app = application

# Load the model and scaler with 'rb' for reading binary files
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get data from form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        region = float(request.form.get('region'))
        Classes = float(request.form.get('Classes'))

        # Create a DataFrame
        input_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, region, Classes]],
                                  columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'region', 'Classes'])

        # Scaling the input data
        new_data = standard_scaler.transform(input_data)

        # Predicting using the ridge model
        result = ridge_model.predict(new_data)
        result = result[0]  # Extract the scalar value

        # Return the result as a string or JSON
        return jsonify({'prediction': float(result)})  # or return str(result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
