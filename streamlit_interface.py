import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Portefeuille Durable 📊🌿", layout='wide')

@st.cache_data
def load_and_clean_data(filepath):
    esg_data = pd.read_excel(filepath, sheet_name='Feuil1')
    # Convert columns to numeric and handle missing values
    cols_to_numeric = ['ESG_SCORE', 'ESG_ENVIRONMENTAL_SCORE', 'ESG_SOCIAL_SCORE', 'ESG_GOVERNANCE_SCORE', 'DIVIDEND_YIELD', 'PE_RATIO']
    for col in cols_to_numeric:
        if col in esg_data.columns:
            esg_data[col] = pd.to_numeric(esg_data[col], errors='coerce')
    esg_data['DIVIDEND_YIELD'] = esg_data['DIVIDEND_YIELD'].fillna(0)
    esg_data = esg_data.dropna(subset=['ESG_SCORE', 'ESG_ENVIRONMENTAL_SCORE', 'ESG_SOCIAL_SCORE', 'ESG_GOVERNANCE_SCORE', 'DIVIDEND_YIELD', 'PE_RATIO'])

    prices = pd.read_excel(filepath, sheet_name='Feuil2', index_col='Dates', parse_dates=True, header=1)
    prices = prices.dropna(axis=1, how='all')
    prices.index = pd.to_datetime(prices.index)
    # Use tickers that are in both ESG data and prices
    esg_data = esg_data[esg_data['Ticker'].isin(prices.columns)]
    return esg_data, prices

def calculate_metrics(prices):
    returns = prices.pct_change().dropna()
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    metrics_df = pd.DataFrame({
        'Ticker': prices.columns,
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatility
    }).set_index('Ticker')
    return metrics_df


def filter_stocks(esg_data, metrics_df, filters, thematic_column_map):
    full_data = esg_data.merge(metrics_df, left_on='Ticker', right_index=True, how='inner')
    
    if 'exclude_sub_industries' in filters and filters['exclude_sub_industries']:
        full_data = full_data[~full_data['GICS_SUB_INDUSTRY_NAME'].isin(filters['exclude_sub_industries'])]
    # Filtre géographique (marché)
    if filters.get('geo_filter'):
        full_data = full_data[full_data['Marché'].isin(filters['geo_filter'])].copy()
    # Filtre sectoriel
    if filters.get('sector_filter'):
        full_data = full_data[full_data['GICS_SECTOR_NAME'].isin(filters['sector_filter'])].copy()
    
    thematic_col = thematic_column_map.get(filters.get('theme', 'Général ESG'), 'ESG_SCORE')
    full_data = full_data.dropna(subset=[thematic_col])

    
    filtered = full_data[
        (full_data['MSCI_ESG_RATING'].isin(filters['esg_rating'])) &
        (full_data['ESG_SCORE'].between(filters['esg_score_min'], filters['esg_score_max'])) &
        (full_data['DIVIDEND_YIELD'] >= filters['div_yield_min']) &
        (full_data['PE_RATIO'] <= filters['pe_ratio_max']) &
        (full_data['Annual Return'] >= filters['return_min']) &
        (full_data['Annual Volatility'] <= filters['vol_max'])
    ]
    return filtered

# Compute the composite score for each stock that meets the chosen criteria, calculate their weights, and retrieve their historical prices
def build_portfolio(filtered, prices, weights_config, start_date, end_date, selected_column='ESG_SCORE', nb_tickers=20):

    # Normalisation dynamique de la colonne ESG sélectionnée
    filtered['ESG_Norm'] = (
        (filtered[selected_column] - filtered[selected_column].min()) /
        (filtered[selected_column].max() - filtered[selected_column].min())
    )

    filtered['Return_Norm'] = (
        (filtered['Annual Return'] - filtered['Annual Return'].min()) /
        (filtered['Annual Return'].max() - filtered['Annual Return'].min())
    )

    filtered['Vol_Norm'] = 1 - (
        (filtered['Annual Volatility'] - filtered['Annual Volatility'].min()) /
        (filtered['Annual Volatility'].max() - filtered['Annual Volatility'].min())
    )

    filtered['Composite_Score'] = (
        weights_config['w_esg'] * filtered['ESG_Norm'] +
        weights_config['w_return'] * filtered['Return_Norm'] +
        weights_config['w_vol'] * filtered['Vol_Norm']
    )

    top_stocks = filtered.sort_values('Composite_Score', ascending=False).head(nb_tickers)
    tickers = top_stocks['Ticker']

    portfolio_prices = prices.loc[start_date:end_date, tickers].dropna(axis=1)

    scores = top_stocks.set_index('Ticker').loc[portfolio_prices.columns, 'Composite_Score']
    weights = scores / scores.sum()

    return portfolio_prices, weights, top_stocks


# Calculate the past performance of the portfolio 
def backtest_performance(portfolio_prices, weights):
    # Calculate daily returns and portfolio returns and cumulative returns
    returns = portfolio_prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    annualized_return = cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    metrics = {
        'Rendement annualisé (%)': annualized_return * 100,
        'Volatilité annualisée (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio
    }

    return cumulative_returns, metrics


# Fonction pour afficher les critères ESG du portefeuille et les secteurs des entreprises sélectionnées
def display_esg_criteria_and_sectors(top_stocks, weights):

    # S'assurer que les index sont alignés
    top_stocks = top_stocks.set_index('Ticker').loc[weights.index].copy()
    # Les critères ESG de notre portefeuille qu'on souhaite afficher 
    esg_criteria = ['ESG_SCORE', 'ESG_ENVIRONMENTAL_SCORE', 'ESG_SOCIAL_SCORE', 'ESG_GOVERNANCE_SCORE']
    # On calcule les critères ESG de notre portefeuille = Somme pondérée des scores du critère ESG de chaque entreprise
    weighted_esg = (top_stocks[esg_criteria].T * weights).T.sum()
    
    # Affichage des scores ESG pondérés
    st.subheader("🌿 Critères ESG pondérés du portefeuille")
    for criterion, value in weighted_esg.items():
        st.write(f"**{criterion}** : {value:.2f}")

    # 🏢 Affichage des informations des entreprises sélectionnées
    st.subheader("🏢 Détails des entreprises sélectionnées")

    display_df = top_stocks[['NAME', 'GICS_SECTOR_NAME', 'Marché']].copy()
    display_df['Poids dans le portefeuille (%)'] = (weights * 100).round(2).values

    display_df = display_df.rename(columns={
        'NAME': 'Nom',
        'GICS_SECTOR_NAME': 'Secteur',
        'Marché': 'Zone géographique'
    })

    # Affichage dans un tableau
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

# Streamlit Interface
st.title("📊 Portefeuille Durable avec ESG 🌿")

filepath = "data_SPX_SXXP.xlsx"
esg_data, prices = load_and_clean_data(filepath)

metrics_df = calculate_metrics(prices)



thematic_column_map = {
    "Général ESG": "ESG_SCORE",
    "Climat": "ESG_ENVIRONMENTAL_SCORE",
    "Social/Éthique": "ESG_SOCIAL_SCORE",
    "Gouvernance": "ESG_GOVERNANCE_SCORE",
    "Accès à l'eau": "WATER_INTENSITY_PER_SALES",
    "Égalité de genre": "BOARD_DIVERSITY_PCT",
    "ODD7 : Énergie propre et abordable": "TOT_GHG_CO2_EM_INTENS_PER_SALES",  # efficacité énergétique
    "ODD13 : Lutte contre le changement climatique": "CDP_CLIMATE_CHANGE_SCORE",         # action climatique
}

# Définir les critères dans le sidebar
with st.sidebar:
    
    st.header("🎯 Thème d'investissement durable")
    selected_theme = st.selectbox("🌱 Choix du thème", list(thematic_column_map.keys()))
    selected_column = thematic_column_map.get(selected_theme, 'ESG_SCORE')

    st.markdown("---")


    # Multiselect pour zone géographique
    geo_options = sorted(esg_data['Marché'].dropna().unique())
    geo_filter = st.multiselect("🌍 Marchés / Zones géographiques", geo_options, default=[])

    # Multiselect pour secteurs
    sector_options = sorted(esg_data['GICS_SECTOR_NAME'].dropna().unique())
    sector_filter = st.multiselect("🏭 Choisir un ou plusieurs secteurs", sector_options, default=[])

    st.markdown("---")

    st.header('Filtres ESG 🌿')
    esg_rating = st.multiselect('MSCI ESG Rating', esg_data['MSCI_ESG_RATING'].dropna().unique(), default=['AA', 'AAA'])
    esg_score_min, esg_score_max = st.slider('Score ESG', 0, 10, (0, 10))

    st.header("🛑 Exclusions ESG")

    # 🔒 Exclusions obligatoires – non modifiables
    forced_exclusions = [
        "Oil & Gas Exploration & Production",
        "Integrated Oil & Gas",
        "Oil & Gas Equipment & Services",
        "Oil & Gas Storage & Transportation",
        "Oil & Gas Refining & Marketing",
        "Tobacco",
        "Fertilizers & Agricultural Chemicals",
        "Diversified Metals & Mining",
        "Copper",
        "Steel"
    ]

    st.markdown("### 🔒 Exclusions ESG obligatoires")
    st.info("Ces sous-industries sont exclues automatiquement pour des raisons ESG fortes (énergies fossiles, pollution, tabac, etc.).")

    for item in forced_exclusions:
        st.markdown(f"- {item}")

    # ⚖️ Exclusions optionnelles – modifiables par l'utilisateur
    optional_esg_exclusions = [
        "Casinos & Gaming",                   # Jeu
        "Passenger Airlines",                # Aviation commerciale
        "Restaurants",                       # Alimentation rapide (junk food, malbouffe)
        "Brewers",                           # Brasseurs
        "Distillers & Vintners",            # Alcools forts
        "Movies & Entertainment",           # Divertissement pas toujours éthique
        "Apparel, Accessories & Luxury",    # Luxe (conditions de travail, consommation ostentatoire)
        "Automobile Manufacturers",         # Impact carbone
        "Air Freight & Logistics",          # Émissions
        "Interactive Home Entertainment",   # Jeux vidéos controversés
    ]

    optional_exclude = st.multiselect(
        "⚖️ Autres secteurs controversés à exclure (optionnel)", 
        optional_esg_exclusions,
        default=["Casinos & Gaming", "Passenger Airlines"]
    )

    # Fusion des exclusions
    exclude_sub_industries = forced_exclusions + optional_exclude






    st.header('Critères financiers 📈')
    div_yield_min = st.slider('Dividend Yield Min (%)', 0.0, 10.0, 1.0)
    pe_ratio_max = st.slider('P/E Ratio Max', 0, 50, 25)

    st.header('Critères historiques 📊')
    return_min = st.slider('Rendement annuel min (%)', -10.0, 50.0, 5.0) / 100
    vol_max = st.slider('Volatilité annuelle max (%)', 5.0, 50.0, 30.0) / 100

    st.header('Pondérations ⚖️')
    #w_esg = st.slider('ESG', 0.0, 1.0, 0.4)
    #w_return = st.slider('Rendement', 0.0, 1.0, 0.4)
    #w_vol = st.slider('Volatilité', 0.0, 1.0, 0.2)

    # L'utilisateur choisit quel critère il veut "fixer"
    fixed_choice = st.selectbox("Critère à fixer", ["ESG", "Rendement", "Volatilité"])

    # Slider pour le critère fixé
    fixed_value = st.slider(f"{fixed_choice} (pondération)", 0.0, 1.0, 0.4)

    # Calcul automatique des deux autres
    remaining = 1.0 - fixed_value
    if fixed_choice == "ESG":
        w_esg = fixed_value
        w_return = remaining / 2
        w_vol = remaining / 2
    elif fixed_choice == "Rendement":
        w_return = fixed_value
        w_esg = remaining / 2
        w_vol = remaining / 2
    else:
        w_vol = fixed_value
        w_esg = remaining / 2
        w_return = remaining / 2

    



# Date de début et de fin pour le backtest
start_date, end_date = st.date_input('Période de Backtest', [pd.to_datetime('2020-01-01'), pd.to_datetime('2025-04-02')]) # pd.to_datetime('today')
filters = {
    'theme': selected_theme,
    'geo_filter': geo_filter,
    'sector_filter': sector_filter,
    'exclude_sub_industries': exclude_sub_industries,
    
    'esg_rating': esg_rating, 
    'esg_score_min': esg_score_min, 
    'esg_score_max': esg_score_max,
    'div_yield_min': div_yield_min, 
    'pe_ratio_max': pe_ratio_max,
    'return_min': return_min, 
    'vol_max': vol_max
    }
weights_config = {
    'w_esg': w_esg, 
    'w_return': w_return, 
    'w_vol': w_vol
    }

filtered_stocks = filter_stocks(esg_data, metrics_df, filters, thematic_column_map)
if not filtered_stocks.empty:
    portfolio_prices, weights, top_stocks = build_portfolio(filtered_stocks, prices, weights_config, start_date, end_date, selected_column)
    st.markdown(f"🧠 Construction du portefeuille optimisé selon le critère : **{selected_theme}**")

    cumulative_returns, metrics = backtest_performance(portfolio_prices, weights)
    # Affichage du portefeuille
    st.subheader("📈 Performance du portefeuille")
    st.line_chart(cumulative_returns)

    st.subheader("📊 Métriques de performance")
    for k, v in metrics.items():
        st.write(f"**{k}** : {v:.2f}")

    # Affichage ESG et secteurs
    display_esg_criteria_and_sectors(top_stocks, weights)

else:
    st.warning("Aucune entreprise ne correspond aux critères sélectionnés.")
