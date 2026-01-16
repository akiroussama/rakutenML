"""
Page Explicabilité des Modèles.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Explicabilité - {APP_CONFIG['title']}",
    page_icon="🔬",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Explicabilité des Modèles")
st.markdown("Comprendre les décisions de l'IA.")

# Méthodes
st.divider()
st.header("Méthodes Utilisées")

col1, col2, col3, col4 = st.columns(4)
col1.metric("SHAP", "Texte")
col2.metric("LIME", "Texte")
col3.metric("Grad-CAM", "Image")
col4.metric("Attention", "BERT")

st.markdown("""
| Méthode | Type | Principe |
|---------|------|----------|
| SHAP | Global/Local | Valeurs de Shapley (théorie des jeux) |
| LIME | Local | Approximation linéaire locale |
| Grad-CAM | Image | Gradients sur dernière couche CNN |
| Attention | Texte | Poids d'attention Transformer |
""")

# SHAP demo
st.divider()
st.header("Exemple SHAP")

st.info("Produit: iPhone 15 Pro Max 256GB Smartphone Apple")

shap_data = pd.DataFrame({
    'Feature': ['iphone', 'smartphone', 'apple', '256gb', 'pro', 'téléphone'],
    'SHAP': [0.42, 0.28, 0.18, 0.12, 0.08, 0.15],
})

fig_shap = px.bar(shap_data, x='SHAP', y='Feature', orientation='h',
                  color='SHAP', color_continuous_scale=['#FFE5E5', '#BF0000'])
fig_shap.update_layout(height=250, coloraxis_showscale=False)
st.plotly_chart(fig_shap, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Prédiction**: Téléphones (2583)")
with col2:
    st.markdown("**Confiance**: 94.2%")

# LIME demo
st.divider()
st.header("Exemple LIME")

st.info("Produit: Console PlayStation 5 jeux vidéo Sony")

lime_data = pd.DataFrame({
    'Word': ['playstation', 'console', 'jeux', 'sony', 'vidéo'],
    'Weight': [0.35, 0.22, 0.18, 0.12, 0.15],
})

fig_lime = px.bar(lime_data, x='Weight', y='Word', orientation='h',
                  color='Weight', color_continuous_scale=['#FFE5E5', '#BF0000'])
fig_lime.update_layout(height=200, coloraxis_showscale=False)
st.plotly_chart(fig_lime, use_container_width=True)

# Métriques
st.divider()
st.header("Métriques")

col1, col2, col3 = st.columns(3)
col1.metric("Fidélité LIME", "R² = 0.89")
col2.metric("Cohérence SHAP", "94%")
col3.metric("Temps/Explication", "2.1s")

# Apport
st.divider()
st.header("Apport")

st.markdown("""
- **Transparence**: Comprendre pourquoi le modèle prédit une catégorie
- **Debugging**: Identifier les erreurs systématiques
- **Confiance**: Renforcer la confiance utilisateur
- **Conformité**: AI Act européen (transparence algorithmique)
""")

# Sidebar
with st.sidebar:
    st.markdown("### Explicabilité")
    st.divider()
    st.markdown("**Méthodes**")
    st.markdown("- SHAP (Texte)")
    st.markdown("- LIME (Texte)")
    st.markdown("- Grad-CAM (Image)")
    st.markdown("- Attention (BERT)")
