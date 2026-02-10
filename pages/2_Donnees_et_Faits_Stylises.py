import streamlit as st

st.title("2. Données & Faits Stylisés")

st.markdown("""
Période d’étude : **2017 – 2026**  
Actifs :
- Crypto : BTC, ETH
- Actions : SPX, NDX
- Macro / Commodities : VIX, US10Y, OIL, GOLD, SILVER
""")

st.image("images/figure1.png", caption="Figure 1 – Log-prix normalisés", use_container_width=True)
st.image("images/figure2.png", caption="Figure 2 – Rendements journaliers", use_container_width=True)
st.image("images/figure3.png", caption="Figure 3 – Matrice de corrélation", use_container_width=True)

st.markdown("""
📌 **Lecture orale**  
- BTC / ETH : cycles violents + drawdowns profonds  
- Corrélation crypto–actions non nulle mais incomplète  
- Justification du travail sur |rₜ| plutôt que rₜ
""")
