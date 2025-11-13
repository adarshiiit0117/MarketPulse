# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
from textblob import TextBlob
import yfinance as yf

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MarketPulse - Unified Financial Dashboard", page_icon="💹", layout="wide")
st.title("💹 MarketPulse - Unified Financial & Sentiment Dashboard")

# ---------------- API KEYS ----------------
tiingo_api_key = "3f69ff9aaf29ca5b8b0daf9340481f323f8481b2"
news_api_key = "aebf914e21f7429d8416bf02c269b305"

default_start_date = "2023-10-16"
default_end_date = "2024-10-16"

# ---------------- Helper Functions ----------------
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def predict_next_day(model, last_data, scaler, time_step):
    last_data = last_data[-time_step:].reshape(1, time_step, 1)
    prediction = model.predict(last_data)
    return scaler.inverse_transform(prediction)

def get_news(query):
    news_url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    response = requests.get(news_url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        st.error("Error fetching news data.")
        return []

def train_lstm(data_scaled, time_step=25):
    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[:train_size], data_scaled[train_size:]
    X_train, y_train = create_dataset(train, time_step)
    X_test, y_test = create_dataset(test, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    test_predict = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_predict)
    return model, mae

# ---------------- MAIN APP ----------------
menu = ["🏠 Welcome", "📈 Stock", "📊 Nifty Index", "₿ Cryptocurrency"]
choice = st.sidebar.selectbox("Select a Section", menu)

# ---------------- WELCOME ----------------
if choice == "🏠 Welcome":
    st.markdown("""
    ## 👋 Welcome to **MarketPulse**
    ### Your all-in-one prediction platform for wealth, markets, and sentiment.
    ---
    🔹 Predict stock, crypto, or index trends using AI (LSTM).  
    🔹 Analyze latest news sentiment automatically.  
    🔹 Make smarter investment decisions with data-driven insights.
    ---
    Select an option from the **sidebar** to begin!
    """)

# ---------------- STOCK ----------------
elif choice == "📈 Stock":
    ticker_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):", value="AAPL").lower()

    url = f"https://api.tiingo.com/tiingo/daily/{ticker_symbol}/prices"
    params = {'startDate': default_start_date, 'endDate': default_end_date, 'token': tiingo_api_key}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        if 'date' not in df.columns:
            st.error("Invalid data format returned from Tiingo.")
        else:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            st.subheader(f"{ticker_symbol.upper()} Stock Prices Over Time")
            st.line_chart(df['close'])

            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df[['close']])
            model, mae = train_lstm(data_scaled)
            predicted_tomorrow = predict_next_day(model, data_scaled, scaler, 25)
            st.success(f"Predicted Closing Price for Tomorrow: ${predicted_tomorrow[0,0]:.2f}")
            st.caption(f"Test MAE: {mae:.5f}")

            st.subheader("📰 Latest News & Sentiment")
            articles = get_news(ticker_symbol)
            for article in articles[:5]:
                title = article['title']
                desc = article.get('description', '')
                sent = TextBlob(desc if desc else title).sentiment
                score = "🟢 Positive" if sent.polarity > 0 else "🔴 Negative" if sent.polarity < 0 else "🟡 Neutral"
                st.markdown(f"### [{title}]({article['url']})")
                st.caption(article['publishedAt'])
                st.write(f"Sentiment: {score}")
                st.write(desc)
                st.write("---")
    else:
        st.error("Error fetching stock data.")

# ---------------- NIFTY INDEX ----------------
elif choice == "📊 Nifty Index":
    st.subheader("Nifty Index Prediction")
    index_choice = st.radio("Select Index", ["Nifty 50", "Nifty 100"])
    index_code = "^NSEI" if index_choice == "Nifty 50" else "^CNX100"

    df = yf.download(index_code, start=default_start_date, end=default_end_date)
    df.reset_index(inplace=True)

    st.line_chart(df['Close'])

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['Close']])
    model, mae = train_lstm(data_scaled)
    predicted_tomorrow = predict_next_day(model, data_scaled, scaler, 25)
    st.success(f"Predicted {index_choice} Closing Price for Tomorrow: ₹{predicted_tomorrow[0,0]:.2f}")
    st.caption(f"Test MAE: {mae:.5f}")

    st.subheader("📰 Latest Nifty News")
    articles = get_news("Nifty 50 OR Nifty 100")
    for article in articles[:5]:
        title = article['title']
        desc = article.get('description', '')
        sent = TextBlob(desc if desc else title).sentiment
        score = "🟢 Positive" if sent.polarity > 0 else "🔴 Negative" if sent.polarity < 0 else "🟡 Neutral"
        st.markdown(f"### [{title}]({article['url']})")
        st.caption(article['publishedAt'])
        st.write(f"Sentiment: {score}")
        st.write(desc)
        st.write("---")

# ---------------- CRYPTO ----------------
elif choice == "₿ Cryptocurrency":
    st.subheader("Cryptocurrency Price Prediction")
    crypto_choice = st.text_input("Enter Crypto ID (e.g., bitcoin, ethereum):", value="bitcoin")

    url = f"https://api.coingecko.com/api/v3/coins/{crypto_choice}/market_chart"
    params = {'vs_currency': 'usd', 'days': '180'}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        prices = data.get('prices', [])
        if not prices:
            st.error("No crypto price data available.")
        else:
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)

            st.line_chart(df['close'])

            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df[['close']])
            model, mae = train_lstm(data_scaled)
            predicted_tomorrow = predict_next_day(model, data_scaled, scaler, 25)
            st.success(f"Predicted {crypto_choice.capitalize()} Price for Tomorrow: ${predicted_tomorrow[0,0]:.2f}")
            st.caption(f"Test MAE: {mae:.5f}")

            st.subheader("📰 Latest Crypto News")
            articles = get_news(crypto_choice)
            for article in articles[:5]:
                title = article['title']
                desc = article.get('description', '')
                sent = TextBlob(desc if desc else title).sentiment
                score = "🟢 Positive" if sent.polarity > 0 else "🔴 Negative" if sent.polarity < 0 else "🟡 Neutral"
                st.markdown(f"### [{title}]({article['url']})")
                st.caption(article['publishedAt'])
                st.write(f"Sentiment: {score}")
                st.write(desc)
                st.write("---")
    else:
        st.error("Error fetching crypto data.")
