import streamlit as st

# --------------------------------------------------
# TITRE
# --------------------------------------------------
st.title("7. Agent IA – Support économétrique")

st.markdown("""
Cette interface s’appuie sur un **agent IA spécialisé en économétrie financière
et en analyse de séries temporelles**, utilisé comme **outil d’assistance méthodologique**
dans le cadre du projet académique :

**« Crypto vs marchés traditionnels : mémoire longue et régimes »**
""")

st.markdown("---")

# --------------------------------------------------
# RÔLE DE L’AGENT
# --------------------------------------------------
st.subheader("Rôle et positionnement de l’agent IA")

st.markdown("""
L’agent IA n’est **ni un outil prédictif**, ni un système d’aide à la décision
en matière d’investissement.

Il agit exclusivement comme :

- un **assistant d’exécution économétrique**,
- un **outil de diagnostic automatique**,
- un **support d’analyse reproductible**.

Son rôle est strictement encadré par une démarche méthodologique définie **a priori**
et validée dans le rapport principal.
""")

# --------------------------------------------------
# CONTRAINTES MÉTHODOLOGIQUES
# --------------------------------------------------
st.subheader("Contraintes méthodologiques strictes")

st.markdown("""
L’agent IA est contraint par les règles suivantes :

- aucune modification de la méthodologie imposée,
- aucun ajout de tests non prévus dans le protocole,
- aucune interprétation hors du cadre académique défini,
- aucune recommandation d’investissement.

Chaque résultat est précédé :
- d’un rappel de la démarche suivie,
- d’une justification économétrique,
- d’une interprétation économique prudente.
""")

# --------------------------------------------------
# CAPACITÉS D’EXÉCUTION
# --------------------------------------------------
st.subheader("Capacités d’exécution")

st.markdown("""
Lorsque l’environnement technique le permet, l’agent est capable de :

- récupérer les données financières (Yahoo Finance),
- calculer les rendements logarithmiques,
- construire les séries |r| et r²,
- exécuter les tests économétriques (BDS, Hurst, stationnarité),
- estimer les modèles ARFIMA et Markov Switching,
- produire des tableaux et visualisations reproductibles.

Dans le cas contraire, il fournit **un code Python complet, documenté
et directement exécutable**, sans jamais inventer de résultats.
""")

# --------------------------------------------------
# LIEN VERS L’AGENT
# --------------------------------------------------
st.subheader("Accès à l’agent IA")

st.markdown("""
L’agent IA est accessible via l’interface suivante :
""")

st.link_button(
    label="Accéder à l’agent IA économétrique",
    url="https://m365.cloud.microsoft:443/chat/?titleId=T_3aa88487-717f-a357-2e02-7f35061efd73&source=embedded-builder"
)

st.markdown("""
⚠️ L’agent est utilisé comme **complément au rapport**,  
et non comme substitut à l’analyse économétrique présentée dans ce travail.
""")

# --------------------------------------------------
# POSITIONNEMENT ACADÉMIQUE
# --------------------------------------------------
st.markdown("---")

st.subheader("Positionnement académique")

st.markdown("""
L’utilisation de l’IA dans ce projet s’inscrit dans une logique de :

- **transparence méthodologique**,
- **reproductibilité des résultats**,
- **assistance technique contrôlée**.

Les choix de modélisation, les hypothèses,
et les interprétations finales restent **entièrement sous la responsabilité
de l’auteur du rapport**.
""")
