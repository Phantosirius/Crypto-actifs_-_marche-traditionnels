import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# TITRE ET CONTEXTE
# ============================================================

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

def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)

    return df


# ============================================================
# VISUALISATIONS
# ============================================================

def run_visualisations() -> None:
    # ------------------------
    # Chargement des données
    # ------------------------
    prices = load_and_clean_csv("data/prices.csv")
    returns = load_and_clean_csv("data/returns_log.csv")
    xls = pd.ExcelFile("data/financial_data.xlsx")

    # ------------------------
    # Aperçu des données
    # ------------------------
    st.subheader("Aperçu des prix — prices.csv")
    st.dataframe(prices.head())

    st.subheader("Aperçu des rendements logarithmiques — returns_log.csv")
    st.dataframe(returns.head())

    # ------------------------
    # Exploration Excel
    # ------------------------
    st.subheader("Structure du fichier Excel — financial_data.xlsx")
    st.write(xls.sheet_names)

    sheet_selected = st.selectbox(
        "Sélectionner une feuille à afficher",
        xls.sheet_names
    )

    df_sheet = pd.read_excel(xls, sheet_name=sheet_selected)
    st.dataframe(df_sheet.head())

    # ========================================================
    # CORRÉLATIONS DYNAMIQUES
    # ========================================================

    required_cols_sp500 = {"Date", "BTC", "SP500"}
    required_cols_vix = {"Date", "BTC", "VIX"}

    # --------------------------------------------------------
    # BTC – SP500
    # --------------------------------------------------------
    st.subheader("Corrélation dynamique BTC – SP500")

    if required_cols_sp500.issubset(returns.columns):
        rolling_corr_sp500 = (
            returns[list(required_cols_sp500)]
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
    else:
        st.warning("Les colonnes BTC et/ou SP500 sont absentes des données.")

    # --------------------------------------------------------
    # BTC – VIX
    # --------------------------------------------------------
    st.subheader("Corrélation dynamique BTC – VIX")

    if required_cols_vix.issubset(returns.columns):
        rolling_corr_vix = (
            returns[list(required_cols_vix)]
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
    else:
        st.warning("Les colonnes BTC et/ou VIX sont absentes des données.")


# ============================================================
# POINT D’ENTRÉE
# ============================================================

if __name__ == "__main__":
    run_visualisations()
