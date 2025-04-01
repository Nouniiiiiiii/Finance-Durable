import yfinance as yf
import pandas as pd
import time
import requests
from io import StringIO

# --- Charger le S&P 500 ---
sp_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
sp_df = pd.read_csv(sp_url)
sp_df["Index"] = "S&P 500"
sp_df["Symbol_clean"] = sp_df["Symbol"]

# --- Charger le STOXX Europe 600 ---
stoxx_url = "https://www.stoxx.com/document/Reports/STOXXSelectionList/2025/March/slpublic_sxxp_20250303.csv"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(stoxx_url, headers=headers)

if response.status_code == 200:
    content = response.content.decode("utf-8")
    stoxx_df = pd.read_csv(StringIO(content), sep=";")
    stoxx_df["Index"] = "STOXX 600"
    print(stoxx_df)
    stoxx_df["Symbol_clean"] = stoxx_df["RIC"] 
else:
    print("Erreur lors du chargement du fichier STOXX.")
    stoxx_df = pd.DataFrame(columns=["Symbol_clean", "Index"])

# --- Combiner les deux ---
combined_df = pd.concat([
    stoxx_df[["Symbol_clean", "Index"]],
    sp_df[["Symbol_clean", "Index"]]
]).drop_duplicates(subset="Symbol_clean")

tickers = combined_df["Symbol_clean"].tolist()
index_map = dict(zip(combined_df["Symbol_clean"], combined_df["Index"]))

# --- Collecte des données ---
results = []

for i, symbol in enumerate(tickers):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sustainability = ticker.sustainability

        info_data = {f"info_{k}": v for k, v in info.items()}
        if sustainability is not None and not sustainability.empty:
            sust_data = sustainability.squeeze().to_dict()
            sust_data = {f"sust_{k}": v for k, v in sust_data.items()}
        else:
            sust_data = {}

        combined_data = {
            "Ticker": symbol,
            "Index": index_map[symbol],
        }
        combined_data.update(info_data)
        combined_data.update(sust_data)

        results.append(combined_data)
        print(f"{i+1}/{len(tickers)} : {symbol} OK")

        time.sleep(0.5)

    except Exception as e:
        print(f"{i+1}/{len(tickers)} : {symbol} ERREUR : {e}")
        continue

# --- Exporter vers CSV ---
df_all = pd.DataFrame(results)
df_all.to_csv("esg_full_data.csv", index=False)
print("Fichier 'esg_full_data.csv' sauvegardé.")