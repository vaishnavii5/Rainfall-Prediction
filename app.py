from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and features
model, model_features = joblib.load('model.pkl')

# Region and month options
regions = [
    "ARUNACHAL PRADESH", "ASSAM & MEGHALAYA", "BIHAR", "CHHATTISGARH", "COASTAL ANDHRA PRADESH",
    "COASTAL KARNATAKA", "EAST MADHYA PRADESH", "EAST RAJASTHAN", "EAST UTTAR PRADESH",
    "GANGETIC WEST BENGAL", "GUJARAT REGION", "HARYANA DELHI & CHANDIGARH", "HIMACHAL PRADESH",
    "JAMMU & KASHMIR", "JHARKHAND", "KERALA", "KONKAN & GOA", "LAKSHADWEEP", "MADHYA MAHARASHTRA",
    "MATATHWADA", "NORTH INTERIOR KARNATAKA", "ORISSA", "PUNJAB", "RAYALSEEMA",
    "SAURASHTRA & KUTCH", "SOUTH INTERIOR KARNATAKA", "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "TAMIL NADU", "TELANGANA", "UTTARAKHAND", "VIDARBHA", "WEST MADHYA PRADESH",
    "WEST RAJASTHAN", "WEST UTTAR PRADESH", "NAGA MANI MIZO TRIPURA"
]
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

def format_region_filename(region):
    return region.lower().replace("&", "and").replace(" ", "_") + ".png"

@app.route('/')
def home():
    return render_template('index.html', regions=regions, months=months)

@app.route('/predict', methods=['POST'])
def predict():
    region = request.form['region']
    month = request.form['month']
    year = int(request.form['year'])

    input_df = pd.DataFrame([[region, year, month]], columns=['SUBDIVISION', 'YEAR', 'MONTH'])
    input_encoded = pd.get_dummies(input_df).reindex(columns=model_features, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    flood_alert = prediction > 300  # mm threshold

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
