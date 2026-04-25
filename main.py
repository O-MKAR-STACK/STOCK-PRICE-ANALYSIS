import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Fetch stock data
stock_symbol = "RELIANCE.NS"
data = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")

data = data[['Close']]
data.dropna(inplace=True)

# Moving averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()

# Prediction column
data['Prediction'] = data['Close'].shift(-10)
data.dropna(inplace=True)

# Prepare data
X = np.array(data[['Close', 'MA50', 'MA100']])
y = np.array(data['Prediction'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

# Buy/Sell signals
data['Signal'] = 0
data.loc[data['MA50'] > data['MA100'], 'Signal'] = 1
data.loc[data['MA50'] < data['MA100'], 'Signal'] = -1

latest_signal = data['Signal'].iloc[-1]

if latest_signal == 1:
    print("\nLatest Signal: BUY")
elif latest_signal == -1:
    print("\nLatest Signal: SELL")
else:
    print("\nLatest Signal: HOLD")

# Plot
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA50'], label='50-Day MA')
plt.plot(data['MA100'], label='100-Day MA')

buy = data[data['Signal'] == 1]
sell = data[data['Signal'] == -1]

plt.scatter(buy.index, buy['Close'], marker='^', label='Buy')
plt.scatter(sell.index, sell['Close'], marker='v', label='Sell')

plt.legend()
plt.grid()
plt.show()