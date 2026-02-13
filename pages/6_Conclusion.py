import streamlit as st

st.title("6. Conclusion générale")

st.markdown("""
## Principaux résultats

Ce travail met en évidence une mémoire longue statistiquement significative dans la
dynamique de la volatilité des crypto-actifs. Malgré leur institutionnalisation,
ces actifs conservent des propriétés statistiques distinctes de celles des marchés
traditionnels.
""")

st.markdown("""
## Implications pour la gestion du risque

L’utilisation de modèles à mémoire courte conduit à une sous-estimation de la
persistance des chocs de volatilité. Les modèles ARFIMA permettent une calibration
plus prudente des mesures de risque, notamment la Value at Risk et l’Expected Shortfall.
""")

st.markdown("""
## Limites et perspectives

Les limites observées à moyen horizon soulignent la nécessité d’extensions
méthodologiques intégrant des changements de régimes ou des dynamiques non
linéaires, telles que les modèles ARFIMA à régimes markoviens ou FIGARCH.
""")
