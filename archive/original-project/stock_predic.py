# import: Stock data from Yahoo Finance
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Download stock data
ticker = "AAPL"
stock_data = yf.download(ticker, start="2010-01-01", end="2025-01-01")

# Feature engineering: Use open, high, low, close, and volume as features
stock_data['Price'] = stock_data['Close']  # target variable (stock price)
features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = stock_data['Price']

# Split Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict stock prices
y_pred = rf.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-Squared: {r2}")

# Cross-validation
cv_scores = cross_val_score(rf, features, target, cv=5, scoring='neg_mean_squared_error')
mean_cv_score = np.mean(np.abs(cv_scores))  # Take the absolute value of negative MSE
print(f"Mean Cross-validation MSE: {mean_cv_score}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 10, 20]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best Score from GridSearchCV: {grid_search.best_score_}")

# Save results to CSV for R
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv("stock_predictions.csv", index=False)

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(y_pred, label="Predicted", color='red')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()