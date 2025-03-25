import streamlit as st

st.title("Hello Streamlit 👋")
st.write("Si tu vois ça, Streamlit fonctionne parfaitement !")


import yfinance as yf

ticker = yf.Ticker("AAPL")  # Apple Inc.

# Données ESG
esg = ticker.sustainability
print(esg)