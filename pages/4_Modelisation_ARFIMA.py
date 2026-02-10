import streamlit as st

st.title("3. Tests économétriques")

st.markdown("""
Tests appliqués :
- Stationnarité (ADF, PP, KPSS)
- Autocorrélation
- Non-linéarité (BDS)
- Exposant de Hurst (DFA)
""")

st.image("images/figure4.png", caption="Figure 4 – Proxys de volatilité |rₜ|", use_container_width=True)
st.image("images/figure5.png", caption="Figure 5 – ACF des proxys de volatilité", use_container_width=True)
st.image("images/figure6.png", caption="Figure 6 – Exposant de Hurst", use_container_width=True)

st.markdown("""
👉 Résultat clé :  
- Décroissance **hyperbolique** des ACF  
- Hurst > 0.5 pour tous les actifs  
➡️ Signature claire de **mémoire longue**
""")
