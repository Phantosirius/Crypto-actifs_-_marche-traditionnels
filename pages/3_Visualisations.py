import streamlit as st
import pandas as pd
import plotly.express as px

st.header("Visualisations et exploration des données")

st.markdown("""
Cette section permet :
- de présenter les données utilisées dans l’analyse,
- d’illustrer les résultats économétriques à l’aide de visualisations,
- sans recalcul en temps réel.
""")

# ============================================================
# FONCTION DE NETTOYAGE DES CSV (ROBUSTE)
# ============================================================

def load_and_clean_csv(path):
    df = pd.read_csv(path)

    # Supprimer colonne d’index parasite
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Renommer la première colonne en Date
    df = df.rename(columns={df.columns[0]: "Date"})

    # Supprimer les lignes non-dates (Ticker, Date, etc.)
    df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]

    # Conversion en datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

prices = load_and_clean_csv("data/prices.csv")
returns = load_and_clean_csv("data/returns_log.csv")

xls = pd.ExcelFile("data/financial_data.xlsx")

# ============================================================
# APERÇU DES DONNÉES
# ============================================================

st.subheader("Aperçu des prix — prices.csv")

st.markdown("""
Ce fichier contient les **prix de clôture** des différents actifs financiers.
Il sert de base au calcul des rendements logarithmiques.
""")

st.dataframe(prices.head())

st.subheader("Aperçu des rendements logarithmiques — returns_log.csv")

st.markdown("""
Ce fichier contient les **rendements logarithmiques journaliers**
calculés à partir des prix de clôture.

Ces rendements sont utilisés pour :
- les statistiques descriptives,
- les tests BDS,
- les modèles multivariés,
- le Markov Switching.
""")

st.dataframe(returns.head())

# ============================================================
# EXPLORATION DU FICHIER EXCEL
# ============================================================

st.subheader("Structure du fichier Excel — financial_data.xlsx")

st.markdown("""
Ce fichier regroupe les différentes séries et transformations
utilisées dans l’analyse, organisées par feuilles.
""")

st.markdown("**Feuilles disponibles :**")
st.write(xls.sheet_names)

sheet_selected = st.selectbox(
    "Sélectionner une feuille à afficher",
    xls.sheet_names
)

df_sheet = pd.read_excel(xls, sheet_name=sheet_selected)
st.dataframe(df_sheet.head())

# ============================================================
# VISUALISATIONS — CORRÉLATIONS DYNAMIQUES
# ============================================================

st.subheader("Corrélation dynamique BTC – SP500")

st.markdown("""
Corrélation de Pearson calculée sur les rendements logarithmiques,
avec une fenêtre glissante de 252 jours.
""")

rolling_corr_sp500 = (
    returns[["Date", "BTC", "SP500"]]
    .dropna()
    .set_index("Date")
    .rolling(window=252)
    .corr()
    .unstack()
    .iloc[:, 1]
    .reset_index()
    .rename(columns={0: "corr"})
)

fig_sp500 = px.line(
    rolling_corr_sp500,
    x="Date",
    y="corr",
    title="Corrélation rolling BTC – SP500 (252 jours)"
)

st.plotly_chart(fig_sp500, use_container_width=True)

# ------------------------------------------------------------

st.subheader("Corrélation dynamique BTC – VIX")

rolling_corr_vix = (
    returns[["Date", "BTC", "VIX"]]
    .dropna()
    .set_index("Date")
    .rolling(window=252)
    .corr()
    .unstack()
    .iloc[:, 1]
    .reset_index()
    .rename(columns={0: "corr"})
)

fig_vix = px.line(
    rolling_corr_vix,
    x="Date",
    y="corr",
    title="Corrélation rolling BTC – VIX (252 jours)"
)

st.plotly_chart(fig_vix, use_container_width=True)
