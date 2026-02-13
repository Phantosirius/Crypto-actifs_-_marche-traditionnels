import streamlit as st

st.title("5. Résultats et interprétation")

st.subheader("Estimation de l’exposant de Hurst")
st.image("images/figure6.png", use_container_width=True)

st.markdown("""
Les estimations obtenues par DFA indiquent une persistance significative de la
volatilité pour l’ensemble des actifs. Les crypto-actifs présentent des exposants
de Hurst supérieurs à 0,5, confirmant l’existence d’une mémoire longue.
""")

st.subheader("Comparaison inter-actifs du paramètre d")
st.image("images/figure9.png", use_container_width=True)

st.markdown("""
Les marchés traditionnels présentent en moyenne des valeurs de d plus élevées,
suggérant des régimes de volatilité plus persistants. Les crypto-actifs se distinguent
par des chocs plus abrupts mais une dissipation plus rapide.
""")

st.subheader("Stabilité temporelle de la mémoire longue")
st.image("images/figure10.png", use_container_width=True)

st.markdown("""
L’instabilité temporelle du paramètre d, particulièrement marquée pour le Bitcoin,
suggère que la mémoire longue observée peut être partiellement induite par des
changements de régimes.
""")
