import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess your updated dataset
df = pd.read_csv('rainfall in india 1901-2015.csv')
df = df.dropna(subset=['STATE_UT_NAME', 'YEAR'])

# Group and average relevant features
monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
df_grouped = df.groupby(['STATE_UT_NAME', 'YEAR'])[monthly_cols].mean().reset_index()

# Define features and target
X = df_grouped[['STATE_UT_NAME', 'YEAR']].copy()
X = pd.get_dummies(X, columns=['STATE_UT_NAME'])  # one-hot encode region
y = df_grouped[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1)  # Jun–Sep monsoon total

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump((model, X.columns.tolist()), 'model.pkl')
