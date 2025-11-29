import streamlit as st
import matplotlib.pyplot as plt
from model.train import train_lstm

st.set_page_config(page_title="Pro Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 Professional LSTM Stock Predictor")
st.write("A high-accuracy forecasting model using LSTM, MACD, RSI, and volatility.")

ticker = st.text_input("Enter stock ticker:", "AAPL")

if st.button("Train & Predict"):
    with st.spinner("Training LSTM model… this may take 20–40 seconds"):
        df, preds, actual = train_lstm(ticker)

    st.success("Training complete!")

    st.subheader("Last 7 Days: Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual[-1], label="Actual (last window)", marker="o")
    ax.plot(preds[-1], label="Predicted (7-day forecast)", marker="o")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📅 Forecasted Prices For Next 7 Days")
    st.write(preds[-1])
