import streamlit as st

st.title("4. Modélisation ARFIMA")

st.markdown("""
## Limites des modèles à mémoire courte

Les modèles ARIMA reposent sur l’hypothèse d’une décroissance exponentielle des
autocorrélations. Cette hypothèse est contredite par l’analyse empirique des proxys
de volatilité, dont l’autocorrélation décroît lentement sur un grand nombre de retards.
""")

st.markdown("""
## Modèle ARFIMA

Le modèle ARFIMA introduit une différenciation fractionnaire permettant de capturer
une dépendance de long terme :

ϕ(L)(1 − L)^d y_t = θ(L)ε_t

Lorsque 0 < d < 0,5, le processus est stationnaire et caractérisé par une mémoire
longue.
""")

st.subheader("Spécification du modèle pivot (Bitcoin)")
st.image("images/figure7.png", use_container_width=True)

st.markdown("""
Le critère d’information bayésien conduit à retenir un modèle ARFIMA(1, d ≈ 0,175, 0)
sur le proxy de volatilité |r_t|.
""")

st.subheader("Diagnostics du modèle")
st.image("images/figure8.png", use_container_width=True)

st.markdown("""
Les tests de Ljung-Box indiquent un bon blanchiment des résidus à court horizon,
mais révèlent une dépendance résiduelle à moyen horizon, suggérant des limites
structurelles du modèle linéaire.
""")
