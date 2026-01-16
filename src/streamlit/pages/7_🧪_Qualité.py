"""
Page Qualité Logicielle & Tests.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Qualité - {APP_CONFIG['title']}",
    page_icon="🧪",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Qualité Logicielle")

# Métriques clés
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tests", "210+")
col2.metric("Couverture", "85%")
col3.metric("Sécurité", "50+")
col4.metric("Tests ML", "40+")
col5.metric("Exécution", "<2min")

# Types de tests
st.divider()
st.header("Types de Tests")

st.markdown("""
| Type | Nombre | Description |
|------|--------|-------------|
| Unitaires | 90 | Fonctions isolées, edge cases |
| Intégration | 30 | Pages, imports, pipeline |
| ML | 40 | Performance, non-régression |
| Sécurité | 50 | XSS, injection, sanitization |
""")

# Couverture
st.divider()
st.header("Couverture de Code")

coverage_data = pd.DataFrame([
    {"Module": "mock_classifier", "Couverture": 92},
    {"Module": "category_mapping", "Couverture": 100},
    {"Module": "preprocessing", "Couverture": 88},
    {"Module": "data_loader", "Couverture": 75},
    {"Module": "image_utils", "Couverture": 82},
    {"Module": "config", "Couverture": 95},
])

fig = px.bar(coverage_data, x='Couverture', y='Module', orientation='h',
             color='Couverture', color_continuous_scale=['#FFE5E5', '#BF0000'],
             range_color=[0, 100])
fig.update_layout(height=300, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Tests ML
st.divider()
st.header("Tests Machine Learning")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Quality Gates")
    st.markdown("""
    - Accuracy >= 75%
    - F1-Score >= 70%
    - Latence < 100ms
    """)

with col2:
    st.subheader("Non-régression")
    st.markdown("""
    - Baseline stocké en JSON
    - Tolérance: 2%
    - Golden predictions
    """)

# Sécurité
st.divider()
st.header("Tests de Sécurité")

st.markdown("""
| Vulnérabilité | Payloads | Status |
|---------------|----------|--------|
| XSS (Script) | 15+ | Bloqué |
| XSS (Events) | 10+ | Bloqué |
| HTML Injection | 8+ | Bloqué |
| Path Traversal | 6+ | Géré |
""")

# Commandes
st.divider()
st.header("Commandes")

st.code("""
# Tous les tests
pytest

# Tests par type
pytest -m unit
pytest -m ml
pytest -m security

# Couverture
pytest --cov=utils --cov-report=html
""", language="bash")

# Sidebar
with st.sidebar:
    st.markdown("### Qualité")
    st.divider()
    st.metric("Tests", "210+")
    st.metric("Couverture", "85%")
    st.divider()
    st.code("pytest -v", language="bash")
