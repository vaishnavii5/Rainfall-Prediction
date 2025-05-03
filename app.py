from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import os

app = Flask(__name__)

# Load dataset
data = pd.read_csv("rainfall in india 1901-2015.csv")

# Use the correct column for all Indian states
states = sorted(data['SUBDIVISION'].unique())

@app.route('/')
def home():
    return render_template("index.html", states=states)

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # Handle AJAX request
        request_data = request.get_json()
        selected_state = request_data.get('state')
        year = int(request_data.get('year', 2025))
        month = int(request_data.get('month', 6))  # Default to June if not provided
        
        # Process prediction
        prediction = process_prediction(selected_state, year)
        
        # Define flood threshold (this is hypothetical and should be adjusted)
        flood_threshold = 300  # mm of rainfall
        
        # Generate flood alert message
        if prediction > flood_threshold:
            flood_alert = "Flood Alert: High risk of flooding! Please take necessary precautions."
        elif prediction > flood_threshold * 0.7:
            flood_alert = "Warning: Moderate risk of flooding possible."
        else:
            flood_alert = "No flood risk predicted at this time."
            
        return jsonify({
            'predicted_rainfall': round(prediction, 2),
            'flood_alert': flood_alert
        })
    else:
        # Handle form submission
        selected_state = request.form['state']
        year = int(request.form['year'])
        
        # Process prediction
        prediction = process_prediction(selected_state, year)
        
        return render_template('result.html', prediction=round(prediction, 2), state=selected_state, year=year)

def process_prediction(selected_state, year):
    # Filter the data for the selected state
    state_data = data[data['SUBDIVISION'] == selected_state]
    
    # Check if state data exists
    if state_data.empty:
        return 0  # Return default value if no data
    
    # Average rainfall by year
    df = state_data.groupby('YEAR').mean(numeric_only=True).reset_index()
    
    # Feature and target
    X = df[['YEAR']]
    y = df[['ANNUAL']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor())
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Make prediction
    prediction = pipeline.predict([[year]])[0]
    
    return prediction

@app.route('/static/maps/<state_name>')
def get_state_map(state_name):
    # This route will serve state map images if needed
    return app.send_static_file(f'img/maps/{state_name}')

if __name__ == '__main__':
    app.run(debug=True)