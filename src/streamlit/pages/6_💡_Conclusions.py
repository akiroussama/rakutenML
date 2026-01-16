"""
Page de conclusions et perspectives.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Conclusions - {APP_CONFIG['title']}",
    page_icon="💡",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Conclusions & Perspectives")

# Résultats
st.divider()
st.header("Résultats")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "85%", "Objectif atteint")
col2.metric("Catégories", "27", "Toutes couvertes")
col3.metric("Meilleur modèle", "CamemBERT")

st.success("Classification automatique de 84 916 produits avec 6 modèles comparés.")

# Impact business
st.divider()
st.header("Impact Business")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Avant (Manuel)")
    st.markdown("""
    - Temps: ~5 min/produit
    - Erreur: 10-15%
    - Scalabilité: Limitée
    """)

with col2:
    st.subheader("Après (IA)")
    st.markdown("""
    - Temps: <1 sec/produit
    - Erreur: ~15%
    - Scalabilité: 100K+/jour
    """)

# Limites
st.divider()
st.header("Limites")

st.markdown("""
| Limite | Impact |
|--------|--------|
| Classes minoritaires | F1 plus faible (~60%) |
| Image seule | ~72% accuracy max |
| Qualité texte | Dépendance aux descriptions |
| Drift temporel | Réentraînement nécessaire |
""")

# Perspectives
st.divider()
st.header("Perspectives")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Court terme")
    st.markdown("""
    - Data augmentation
    - Ensemble learning
    - Seuil de confiance
    """)

with col2:
    st.subheader("Moyen terme")
    st.markdown("""
    - Fine-tuning CamemBERT
    - Modèle CLIP
    - Active learning
    """)

with col3:
    st.subheader("MLOps")
    st.markdown("""
    - Pipeline CI/CD
    - Monitoring drift
    - A/B testing
    """)

# Conclusion
st.divider()
st.header("Conclusion")

st.info("""
**Mission accomplie**: Classification automatique de produits e-commerce avec 85% d'accuracy.
Solution scalable, maintenable et prête pour la production.
""")

# Sidebar
with st.sidebar:
    st.markdown("### Conclusions")
    st.divider()
    st.success("Accuracy: 85%")
    st.success("27 catégories")
    st.success("6 modèles comparés")
    st.divider()
    st.markdown("**Remerciements**")
    st.markdown("DataScientest, Mentors, Équipe")
