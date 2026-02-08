import streamlit as st

st.set_page_config(
    page_title="Crypto vs Marchés Traditionnels",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Crypto vs marchés traditionnels : mémoire longue et régimes")

st.markdown("""
Cette application Streamlit accompagne un projet académique portant sur la comparaison
des dynamiques des cryptomonnaies et des marchés financiers traditionnels.

L’analyse économétrique est réalisée en amont.
Cette interface a pour objectif de **présenter**, **visualiser** et **documenter**
les résultats de manière structurée et interactive.
""")

st.info(
    "Navigation via le menu à gauche. "
    "Les résultats complets sont disponibles dans la page **Rapport**."
)
