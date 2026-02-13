import streamlit as st

st.title("1️⃣ Introduction")

st.markdown("""
## 🌍 Contexte économique et financier

L’émergence des crypto-actifs, initiée par le Bitcoin en 2009, a profondément transformé
le paysage financier mondial. Initialement perçus comme des actifs marginaux,
ils se sont progressivement intégrés aux marchés financiers traditionnels,
notamment à travers l’apparition des **ETF spot** sur Bitcoin et Ethereum.

Cette institutionnalisation marque une nouvelle phase, souvent qualifiée de
**« wall-streetisation »**, caractérisée par :
- une augmentation massive de la liquidité,
- l’entrée d’investisseurs institutionnels,
- une interconnexion accrue avec les marchés traditionnels.
""")

st.markdown("""
## ❓ Problématique centrale

Cette intégration soulève une question économétrique fondamentale :

**Les crypto-actifs ont-ils perdu leur caractère fractal et non linéaire,
ou bien ces propriétés persistent-elles malgré leur maturité croissante ?**
""")

st.markdown("""
Plus précisément, nous cherchons à déterminer :
- si la volatilité des crypto-actifs présente une **mémoire longue**,
- si cette mémoire est plus marquée que sur les marchés traditionnels,
- et quelles sont les implications en matière de **gestion du risque**.
""")

st.markdown("""
## 🧪 Hypothèses de recherche

- **H1** : Les proxys de volatilité du Bitcoin et de l’Ethereum présentent une mémoire longue significative  
- **H2** : Les dynamiques observées sont non linéaires (rejet du test BDS)  
- **H3** : L’institutionnalisation n’a pas éliminé la structure fractale des crypto-actifs
""")

st.markdown("""
## 🛠️ Approche méthodologique

Notre analyse combine :
- faits stylisés et analyse descriptive,
- tests économétriques (stationnarité, non-linéarité, persistance),
- modélisation avancée via les modèles **ARFIMA**.
""")
