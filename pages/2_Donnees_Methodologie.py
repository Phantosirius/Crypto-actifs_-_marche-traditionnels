import streamlit as st

st.header("Données et méthodologie")

st.subheader("Sources de données")
st.markdown("""
- Données financières issues de **Yahoo Finance**
- Séries : BTC, ETH, SP500, NASDAQ, VIX, GOLD, DXY, US10Y
- Fréquence quotidienne
- Période : 2015–2024
""")

st.subheader("Prétraitement et alignement")
st.markdown("""
Les séries présentent des calendriers hétérogènes
(crypto 7j/7, marchés traditionnels jours ouvrés).

Les analyses sont réalisées sur des **sous-échantillons alignés**,
obtenus par exclusion des dates pour lesquelles au moins une série est manquante.
Aucune imputation n’est effectuée afin d’éviter l’introduction de dépendances artificielles.
""")

st.subheader("Méthodes économétriques")
st.markdown("""
- Tests de non-linéarité (BDS)
- Exposant de Hurst (|r|, r²)
- Tests de stationnarité (ADF, KPSS)
- ARFIMA (volatilité)
- Markov Switching (rendements BTC)
- Régressions multivariées et corrélations dynamiques
""")
