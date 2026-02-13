import streamlit as st

st.title("1. Introduction")

st.markdown("""
## Contexte général

L’intégration progressive des crypto-actifs dans le système financier international
constitue l’un des phénomènes majeurs des marchés financiers contemporains.
Initialement perçus comme des actifs spéculatifs marginaux, le Bitcoin et l’Ethereum
ont progressivement attiré des flux institutionnels significatifs, notamment à travers
le lancement des ETF spot.

Cette institutionnalisation a profondément modifié la microstructure des marchés
crypto, en augmentant la liquidité, en réduisant certaines inefficiences et en
renforçant les interconnexions avec les marchés traditionnels.
""")

st.markdown("""
## Problématique

D’un point de vue économétrique, cette évolution soulève une question centrale :
les propriétés statistiques spécifiques aux crypto-actifs, telles que la non-linéarité
et la mémoire longue de la volatilité, ont-elles été atténuées par leur maturité
croissante ou persistent-elles malgré l’intégration institutionnelle ?
""")

st.markdown("""
## Objectifs de l’étude

L’objectif de ce travail est double :
- analyser la persistance de la volatilité des crypto-actifs,
- comparer cette persistance à celle observée sur les marchés traditionnels.

L’enjeu est d’évaluer la pertinence des modèles à mémoire longue pour la gestion
du risque sur les marchés crypto-financiers.
""")

st.markdown("""
## Hypothèses testées

- H1 : les proxys de volatilité du Bitcoin et de l’Ethereum présentent une mémoire longue significative ;
- H2 : les dynamiques observées sont non linéaires, même après filtrage ARMA-GARCH ;
- H3 : l’institutionnalisation des crypto-actifs n’a pas éliminé leur structure fractale.
""")

st.markdown("""
## Démarche méthodologique

L’analyse repose sur une combinaison de faits stylisés, de tests économétriques
(stationnarité, non-linéarité, persistance) et de modèles ARFIMA, conformément
à la démarche de Box-Jenkins étendue à la mémoire longue.
""")
