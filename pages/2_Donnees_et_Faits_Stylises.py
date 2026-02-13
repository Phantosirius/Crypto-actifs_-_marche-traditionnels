import streamlit as st

st.title("2. Données et faits stylisés")

st.markdown("""
## Données

L’étude repose sur un panel de dix actifs couvrant différentes classes de marchés :
- crypto-actifs : Bitcoin (BTC), Ethereum (ETH) ;
- marchés actions : S&P 500 (SPX), Nasdaq (NDX) ;
- indicateurs macro-financiers : VIX, DXY, taux US à 10 ans ;
- matières premières : or, pétrole, argent.
""")

st.markdown("""
La période étudiée s’étend de novembre 2017 à 2026, permettant d’inclure plusieurs
cycles financiers, des épisodes de stress systémique et la phase récente
d’institutionnalisation des crypto-actifs.
""")

st.markdown("---")

st.subheader("Dynamique des prix")
st.image("images/figure1.png", use_container_width=True)

st.markdown("""
Les log-prix normalisés mettent en évidence des trajectoires de long terme très
différenciées. Les crypto-actifs présentent une croissance cumulée marquée,
ponctuée de phases de correction importantes, tandis que les indices actions
affichent des dynamiques plus régulières.
""")

st.subheader("Rendements journaliers")
st.image("images/figure2.png", use_container_width=True)

st.markdown("""
Les rendements journaliers révèlent une forte hétérogénéité des amplitudes et la
présence de chocs extrêmes fréquents sur les crypto-actifs. Certains actifs
traditionnels, tels que le VIX ou le pétrole, présentent toutefois des épisodes de
volatilité comparables.
""")

st.subheader("Corrélations entre actifs")
st.image("images/figure3.png", use_container_width=True)

st.markdown("""
La matrice de corrélation montre une forte corrélation entre Bitcoin et Ethereum,
ainsi qu’une corrélation modérée avec les marchés actions, suggérant une
intégration partielle mais non complète des crypto-actifs au système financier
traditionnel.
""")

st.subheader("Proxys de volatilité")
st.image("images/figure4.png", use_container_width=True)

st.markdown("""
La volatilité absolue des rendements met en évidence un phénomène marqué de
volatility clustering et des régimes persistants de stress, justifiant l’étude de la
mémoire longue.
""")
