"""
Page de Performance du Modèle.
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
from utils.category_mapping import CATEGORY_MAPPING, CATEGORY_CODES
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Performance - {APP_CONFIG['title']}",
    page_icon="📈",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Données mock
@st.cache_data
def get_metrics():
    np.random.seed(42)
    global_metrics = {
        "accuracy": 0.847, "f1_macro": 0.823, "f1_weighted": 0.851,
        "precision": 0.835, "recall": 0.812
    }

    category_metrics = []
    for code in CATEGORY_CODES:
        name, _, emoji = CATEGORY_MAPPING[code]
        f1 = np.random.uniform(0.70, 0.95)
        category_metrics.append({
            "code": code, "name": name, "emoji": emoji,
            "f1": min(f1, 0.98),
            "precision": min(f1 + np.random.uniform(-0.05, 0.08), 0.99),
            "recall": min(f1 + np.random.uniform(-0.08, 0.05), 0.97),
            "support": np.random.randint(800, 4500)
        })
    return global_metrics, pd.DataFrame(category_metrics)

@st.cache_data
def get_confusion_matrix():
    np.random.seed(42)
    n = 27
    cm = np.zeros((n, n))
    for i in range(n):
        cm[i, i] = np.random.randint(700, 3500)
        for j in np.random.choice([x for x in range(n) if x != i], 3, replace=False):
            cm[i, j] = np.random.randint(10, 150)
    return cm.astype(int)

global_metrics, category_df = get_metrics()
confusion_matrix = get_confusion_matrix()

# Header
st.title("Performance du Modèle")
st.info("Données de démonstration")

# Métriques globales
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{global_metrics['accuracy']:.1%}")
col2.metric("F1 Macro", f"{global_metrics['f1_macro']:.1%}")
col3.metric("F1 Weighted", f"{global_metrics['f1_weighted']:.1%}")
col4.metric("Precision", f"{global_metrics['precision']:.1%}")
col5.metric("Recall", f"{global_metrics['recall']:.1%}")

# Matrice de confusion
st.divider()
st.header("Matrice de Confusion")

normalize = st.checkbox("Normaliser (%)", value=True)
labels = [CATEGORY_MAPPING[code][0][:8] for code in CATEGORY_CODES]

cm_display = confusion_matrix.astype(float)
if normalize:
    cm_display = cm_display / cm_display.sum(axis=1, keepdims=True) * 100

fig_cm = go.Figure(data=go.Heatmap(
    z=cm_display, x=labels, y=labels,
    colorscale=[[0, '#FFFFFF'], [0.5, '#FFB4B4'], [1, '#BF0000']],
    text=np.round(cm_display, 1 if normalize else 0),
    texttemplate="%{text}", textfont={"size": 7}
))
fig_cm.update_layout(height=600, xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                     yaxis=dict(tickfont=dict(size=8)))
st.plotly_chart(fig_cm, use_container_width=True)

# Performance par catégorie
st.divider()
st.header("Performance par Catégorie")

sorted_df = category_df.sort_values("f1", ascending=False)

fig_cat = px.bar(sorted_df, x=[f"{r['emoji']} {r['name']}" for _, r in sorted_df.iterrows()],
                 y="f1", color="f1", color_continuous_scale=['#FFE5E5', '#BF0000'],
                 labels={"y": "F1-Score", "x": "Catégorie"})
fig_cat.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                      xaxis=dict(tickangle=45, tickfont=dict(size=9)))
st.plotly_chart(fig_cat, use_container_width=True)

# Top/Flop
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 5")
    for _, r in category_df.nlargest(5, 'f1').iterrows():
        st.markdown(f"{r['emoji']} **{r['name']}**: {r['f1']:.1%}")

with col2:
    st.subheader("A améliorer")
    for _, r in category_df.nsmallest(5, 'f1').iterrows():
        st.markdown(f"{r['emoji']} **{r['name']}**: {r['f1']:.1%}")

# Comparaison modalités
st.divider()
st.header("Comparaison des Modalités")

st.markdown("""
| Modalité | Accuracy | F1-Score |
|----------|----------|----------|
| Texte seul | 79.5% | 76.2% |
| Image seule | 72.8% | 68.5% |
| **Multimodal** | **84.7%** | **82.3%** |
""")

# Sidebar
with st.sidebar:
    st.markdown("### Performance")
    st.divider()
    st.metric("Accuracy", f"{global_metrics['accuracy']:.1%}")
    st.metric("F1-Score", f"{global_metrics['f1_weighted']:.1%}")
    st.divider()
    st.download_button("Export CSV", category_df.to_csv(index=False),
                       "metrics.csv", "text/csv")
