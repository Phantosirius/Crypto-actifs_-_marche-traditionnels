import streamlit as st

st.header("Présentation du projet")

st.markdown("""
### Sujet
**Crypto vs marchés traditionnels : mémoire longue et régimes**

### Objectif
Comparer les cryptomonnaies (principalement Bitcoin) et les marchés traditionnels
sous trois angles :
- mémoire longue,
- non-linéarité,
- régimes de marché.

### Démarche générale
1. Analyse descriptive des données
2. Tests préliminaires (BDS, Hurst, stationnarité)
3. Modélisation (ARFIMA, Markov Switching, multivariée)
4. Discussion économique

L’outil Streamlit est utilisé comme **support de restitution**.
""")
