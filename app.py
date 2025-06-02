from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

data = joblib.load('model.pkl')
model = data['pipeline']  # Using the complete pipeline which includes preprocessing
features = data['features']

form_fields = [
    {'name': 'number of bedrooms', 'type': 'number', 'label': 'Number of Bedrooms', 'min': '1', 'max': '10'},
    {'name': 'number of bathrooms', 'type': 'number', 'label': 'Number of Bathrooms', 'min': '1', 'max': '10'},
    {'name': 'living area', 'type': 'number', 'label': 'Living Area (sq ft)', 'min': '100'},
    {'name': 'lot area', 'type': 'number', 'label': 'Lot Area (sq ft)', 'min': '0'},
    {'name': 'number of floors', 'type': 'number', 'label': 'Number of Floors', 'min': '1', 'max': '4'},
    {'name': 'waterfront present', 'type': 'select', 'label': 'Waterfront Present', 'options': ['0', '1']},
    {'name': 'number of views', 'type': 'number', 'label': 'Number of Views', 'min': '0', 'max': '10'},
    {'name': 'condition of the house', 'type': 'number', 'label': 'Condition of the House (1-5)', 'min': '1', 'max': '5'},
    {'name': 'grade of the house', 'type': 'number', 'label': 'Grade of the House (1-13)', 'min': '1', 'max': '13'},
    {'name': 'Area of the house(excluding basement)', 'type': 'number', 'label': 'House Area (excl. basement) sq ft', 'min': '0'},
    {'name': 'Area of the basement', 'type': 'number', 'label': 'Basement Area (sq ft)', 'min': '0'},
    {'name': 'Built Year', 'type': 'number', 'label': 'Built Year', 'min': '1900', 'max': '2025'},
    {'name': 'Renovation Year', 'type': 'number', 'label': 'Renovation Year (if any)', 'min': '1900', 'max': '2025'},
    {'name': 'Postal Code', 'type': 'number', 'label': 'Postal Code', 'min': '0'},
    {'name': 'Number of schools nearby', 'type': 'number', 'label': 'Number of Schools Nearby', 'min': '0'},
    {'name': 'Distance from the airport', 'type': 'number', 'label': 'Distance from Airport (km)', 'min': '0'}
]

df = pd.read_csv('House Price India.csv')
medians = df.median(numeric_only=True)
def prepare_input(form):
    input_data = []
    # Order must match the features used in the model
    for feat in features:
        if feat in [f['name'] for f in form_fields]:
            field = next(f for f in form_fields if f['name'] == feat)
            if field['type'] == 'number':
                value = float(form.get(field['name'], 0))
                input_data.append(value)
            elif field['type'] == 'select':
                value = form.get(field['name'])
                input_data.append(int(value))
        else:
            # Use median for unused features
            input_data.append(medians[feat] if feat in medians else 0)
    return np.array([input_data])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        arr = prepare_input(request.form)
        prediction = model.predict(arr)[0]  # Using the complete pipeline for prediction
        return render_template('result.html', prediction=prediction)
    return render_template('index.html', form_fields=form_fields)

if __name__ == '__main__':
    app.run(debug=True)
