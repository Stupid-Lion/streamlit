import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# App title
st.title("주식 예측 웹 애플리케이션")

# Display popular stock tickers as a table
stock_data = {
    "회사": ["Apple", "Tesla", "Google", "Amazon", "Microsoft", "NVIDIA", "Meta (Facebook)", "Netflix", "AMD", "Disney"],
    "주식 티커": ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "NVDA", "META", "NFLX", "AMD", "DIS"]
}

# Convert to DataFrame
stock_df = pd.DataFrame(stock_data)

# Display stock ticker list as a table
st.write("인기 있는 주식 티커 목록:")
st.dataframe(stock_df)

# Get stock ticker from user
ticker = st.text_input("원하는 주식 티커를 입력하세요 (예: AAPL, TSLA)", "AAPL")

# Set the start and end dates for the data
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

# Download stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Sort data in reverse order (show most recent data first)
data = data.sort_index(ascending=False)

# Display the data if the checkbox is checked
if st.checkbox('데이터 확인'):
    st.write(data)

# Use closing price to make predictions
X = np.array(range(len(data))).reshape(-1, 1)  # Date sequence
y = data['Close'].values  # Closing prices

# Linear regression model for prediction
model = LinearRegression()
model.fit(X, y)

# Generate predictions
predictions = model.predict(X)

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.plot(data.index, y, label="실제 종가", color='blue')
plt.plot(data.index, predictions, label="예측 종가", color='red', linestyle='--')
plt.title(f"{ticker} Stock Prediction")  # Changed title to English
plt.xlabel("Date")  # Changed x-axis label to English
plt.ylabel("Price")  # Changed y-axis label to English
plt.legend()
st.pyplot(plt)

# Predict the next 7 days
future_dates = np.array(range(len(data), len(data) + 7)).reshape(-1, 1)
future_predictions = model.predict(future_dates)

st.write(f"미래 7일 예측 (다음 7일의 주식 가격):")
for i in range(7):
    st.write(f"{(datetime.today() + timedelta(days=i+1)).strftime('%Y-%m-%d')}: {future_predictions[i].item():.2f} USD")
