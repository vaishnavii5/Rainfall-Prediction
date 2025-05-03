from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and feature columns
model, model_features = joblib.load('model.pkl')

# Define region and month options
regions = [
    'ANDAMAN & NICOBAR ISLANDS', 'ARUNACHAL PRADESH', 'ASSAM & MEGHALAYA',
    'NAGA MANI MIZO TRIPURA', 'SUB HIMALAYAN WEST BENGAL & SIKKIM',
    'GANGETIC WEST BENGAL', 'ORISSA', 'JHARKHAND', 'BIHAR', 'EAST UTTAR PRADESH',
    'WEST UTTAR PRADESH', 'UTTARAKHAND', 'HARYANA DELHI & CHANDIGARH',
    'PUNJAB', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'WEST RAJASTHAN',
    'EAST RAJASTHAN', 'WEST MADHYA PRADESH', 'EAST MADHYA PRADESH',
    'GUJARAT REGION', 'SAURASHTRA & KUTCH', 'KONKAN & GOA', 'MADHYA MAHARASHTRA',
    'MARATHWADA', 'VIDARBHA', 'CHHATTISGARH', 'COASTAL ANDHRA PRADESH',
    'TELANGANA', 'RAYALASEEMA', 'TAMIL NADU', 'COASTAL KARNATAKA',
    'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA', 'KERALA',
    'LAKSHADWEEP'
]
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Helper to format map filename
def format_region_filename(region_name):
    return region_name.lower().replace(" & ", "_").replace(" ", "_") + '.png'

@app.route('/')
def index():
    return render_template('index.html', regions=regions, months=months)

@app.route('/predict', methods=['POST'])
def predict():
    region = request.form['region']
    month = request.form['month']
    year = int(request.form['year'])

    # Create input dataframe
    input_df = pd.DataFrame([[region, year, month]], columns=['SUBDIVISION', 'YEAR', 'MONTH'])

    # One-hot encode to match training
    input_encoded = pd.get_dummies(input_df).reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]

    # Example threshold for flood alert
    flood_alert = prediction > 300

    map_filename = format_region_filename(region)

    return render_template(
        'index.html',
        prediction=round(prediction, 2),
        flood_alert=flood_alert,
        regions=regions,
        months=months,
        selected_region=region,
        selected_month=month,
        selected_year=year,
        map_filename=map_filename
    )

if __name__ == '__main__':
    app.run(debug=True)
