import streamlit as st

st.header("Rapport académique")

st.markdown("""
Cette page contient le **rapport complet**, structuré selon le plan académique :

I. Introduction  
II. Données et statistiques descriptives  
III. Tests préliminaires  
IV. Modélisation  
V. Discussion et interprétation économique  
VI. Conclusion et limites  

Les sections précédentes de l’application servent de support
de lecture et de visualisation.
""")

with open("assets/report.pdf", "rb") as f:
    st.download_button(
        label="📄 Télécharger le rapport complet (PDF)",
        data=f,
        file_name="Crypto_vs_Marches_Traditionnels.pdf",
        mime="application/pdf"
    )
