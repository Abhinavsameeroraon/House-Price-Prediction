import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("../data/house_prices.csv")

# Preprocessing
X = df[['Area (sqft)', 'Bedrooms']]
y = df['Price (₹)']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "../models/house_price_model.pkl")
print("Model saved successfully!")
