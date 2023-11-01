import pandas as pd
import numpy as nppop
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle 

data = pd.read_csv("forestfires.csv")
target_column = "area"
drop_features = [target_column, 'X', 'Y', 'rain', 'month', 'day']
features = data.drop(drop_features, axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, data[target_column], test_size=0.1, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error (MSE) to evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mae}")

pickle.dump(model, open('model.pkl','wb'))