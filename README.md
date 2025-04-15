# 🌿 GreenVest - Plateforme d'Investissement Responsable

Bienvenue dans notre projet de finance durable réalisé dans le cadre du cours Finance Durable de l’Université Paris Dauphine - PSL.

## 📌 À propos
GreenVest est une plateforme web interactive qui permet à tout investisseur de construire un portefeuille aligné avec ses valeurs, en intégrant les critères extra-financiers ESG.

## 🎯 Objectifs du projet
Permettre à tout utilisateur de construire un portefeuille personnalisé, en combinant performance, impact environnemental, et valeurs éthiques. Grâce à une interface claire et interactive, GreenVest vous aide à :
- Visualiser l'impact des critères ESG dans la constitution d’un portefeuille
- Combiner performance financière et cohérence éthique
- Intégrer des exclusions sectorielles, des filtrages personnalisés
- Proposer un backtest des performances historiques
- Rendre accessible la construction responsable de portefeuilles

## 🌱 Thèmes d’investissement disponibles
L’utilisateur peut choisir un axe prioritaire parmi 8 thèmes. Chacun active un critère spécifique dans les données ESG :

| Thème                                 | Description                                                    | 
|--------------------------------------|----------------------------------------------------------------|
| Général ESG                           | Approche globale sur les trois piliers ESG                     | 
| Climat                                | Score environnemental                                          | 
| Social / Éthique                      | Score social                                                   | 
| Gouvernance                           | Qualité de la gouvernance d'entreprise                         | 
| Accès à l'eau                         | Intensité d’usage de l’eau par chiffre d’affaires              | 
| Égalité de genre                      | Part de femmes au conseil d’administration                     | 
| ODD 7 : Énergie propre et abordable  | Efficacité énergétique / émissions de CO2 par CA               | 
| ODD 13 : Lutte contre le climat       | Score CDP sur l’action climatique  


## 🧠 Construction du portefeuille
Le portefeuille est construit automatiquement à partir d’une approche combinant critères ESG, financiers et historiques, selon les préférences définies par l’utilisateur.

### Étapes de construction :

1. **Filtrage des entreprises**  
   L’utilisateur commence par appliquer des filtres :
   - Exclusion automatique des sous-industries controversées (énergies fossiles, tabac, métaux, etc.)
   - Possibilité d’exclure d’autres secteurs sensibles (luxe, aviation, jeux, etc.)
   - Sélection géographique (zones de marché) et sectorielle
   - Paramètres ESG : note ESG minimale, rating MSCI
   - Critères financiers : P/E ratio maximal, dividend yield minimal
   - Critères historiques : rendement annuel minimal, volatilité maximale

2. **Choix du thème ESG prioritaire**  
   L’utilisateur sélectionne un axe d’investissement durable (par exemple : Climat, Égalité de genre, Gouvernance...). Ce choix détermine le score ESG utilisé pour évaluer les entreprises.

3. **Évaluation des entreprises**  
   Chaque entreprise restante est analysée selon trois dimensions :
   - Sa performance sur le critère ESG choisi
   - Son rendement annualisé (à partir des prix historiques)
   - Sa volatilité historique

   Ces trois critères sont normalisés pour être comparables, puis pondérés selon les priorités fixées par l’utilisateur (ESG, rendement ou risque).

4. **Sélection et pondération du portefeuille**  
   Les 20 entreprises les mieux classées selon ce score global sont sélectionnées. Leur poids dans le portefeuille est proportionnel à leur score composite, ce qui met en avant les titres les plus alignés avec les préférences définies.

5. **Analyse de la performance**  
   L’application effectue un backtest sur la période choisie :
   - Rendement cumulé
   - Volatilité glissante (21 jours)
   - Max drawdown
   - Sharpe ratio

6. **Affichage graphique**  
   L’utilisateur peut visualiser :
   - La répartition sectorielle et géographique du portefeuille (diagrammes circulaires)
   - Les scores ESG pondérés par entreprise (barres verticales)
   - Les critères ESG agrégés à l’échelle du portefeuille

7. **Résumé téléchargeable**  
   Un résumé des performances et des caractéristiques du portefeuille pourra être exporté au format PDF.


## 🌐 Lien vers l'application
https://finance-durable-ky8tcma3dmqogls5urc4nx.streamlit.app


