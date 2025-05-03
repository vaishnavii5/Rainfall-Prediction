import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and clean dataset
df = pd.read_csv('rainfall in india 1901-2015.csv')
df = df.dropna(subset=['SUBDIVISION', 'YEAR'])

# Reshape data for month-wise predictions
monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
df_melted = df.melt(id_vars=['SUBDIVISION', 'YEAR'], 
                    value_vars=monthly_cols,
                    var_name='MONTH', value_name='RAINFALL')

# Remove rows with missing rainfall values
df_melted.dropna(subset=['RAINFALL'], inplace=True)

# Prepare features
X = df_melted[['SUBDIVISION', 'YEAR', 'MONTH']]
X = pd.get_dummies(X)
y = df_melted['RAINFALL']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump((model, X.columns.tolist()), 'model.pkl')