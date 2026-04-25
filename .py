# -------------------------------
# Required Libraries
# Install using:
# pip install yfinance pandas numpy matplotlib scikit-learn
# -------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Fetch Stock Data
# -------------------------------
stock_symbol = "RELIANCE.NS"   # Change to any stock
data = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")

# Keep only Close price
data = data[['Close']]
data.dropna(inplace=True)

# -------------------------------
# 2. Feature Engineering
# -------------------------------
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()

# Predict future prices (next 10 days)
data['Prediction'] = data['Close'].shift(-10)

data.dropna(inplace=True)

# -------------------------------
# 3. Prepare Data
# -------------------------------
X = np.array(data[['Close', 'MA50', 'MA100']])
y = np.array(data['Prediction'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. Model Evaluation
# -------------------------------
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

# -------------------------------
# 6. Future Predictions
# -------------------------------
future_days = 10
future_data = X[-future_days:]
future_predictions = model.predict(future_data)

print("\nFuture Predictions:")
for i, price in enumerate(future_predictions):
    print(f"Day {i+1}: {price:.2f}")

# -------------------------------
# 7. Visualization
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA50'], label='50-Day MA')
plt.plot(data['MA100'], label='100-Day MA')

plt.title(f"{stock_symbol} Stock Analysis")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()

plt.show()