import streamlit as st
import os

# --------------------------------------------------
# CONFIG PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="Crypto-actifs & Marchés Traditionnels",
    layout="wide"
)

# --------------------------------------------------
# TITRE
# --------------------------------------------------
st.title("Crypto-actifs & Marchés Traditionnels")
st.subheader("Persistance de la volatilité et mémoire longue")

st.markdown("""
Présentation interactive du rapport académique  
**Crypto-actifs vs Marchés traditionnels**  
""")

st.markdown("---")

# --------------------------------------------------
# CHEMIN ABSOLU SÉCURISÉ
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "Crypto_actifs_vs_Marchés_Traditionnels___Persistance_de_la_Volatilité_et_Mémoire_Longue.pdf"
)

# --------------------------------------------------
# TÉLÉCHARGEMENT PDF
# --------------------------------------------------
if os.path.exists(PDF_PATH):
    with open(PDF_PATH, "rb") as f:
        st.download_button(
            label="📄 Télécharger le rapport complet (PDF)",
            data=f,
            file_name="rapport_crypto_volatilite.pdf",
            mime="application/pdf"
        )
else:
    st.error(f"❌ Fichier introuvable : {PDF_PATH}")

st.markdown("""
👉 Utilise le menu à gauche pour naviguer dans la présentation.
""")
