"""
Page de comparaison des modèles de classification.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.mock_classifier import (
    MultiModelClassifier,
    TEXT_MODELS,
    IMAGE_MODELS,
    get_available_text_models,
    get_available_image_models,
)
from utils.category_mapping import get_category_info
from utils.image_utils import load_image_from_upload, validate_image
from utils.preprocessing import preprocess_product_text
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Modèles - {APP_CONFIG['title']}",
    page_icon="🧠",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Session state
if "multi_model_classifier" not in st.session_state:
    st.session_state.multi_model_classifier = MultiModelClassifier()
if "model_comparison_results" not in st.session_state:
    st.session_state.model_comparison_results = None
if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = "text"

# Header
st.title("Comparaison des Modèles")
st.markdown("Comparez les 3 modèles texte ou image sur la même entrée.")

# Mode selection
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("Modèles Texte", use_container_width=True,
                 type="primary" if st.session_state.comparison_mode == "text" else "secondary"):
        st.session_state.comparison_mode = "text"
        st.session_state.model_comparison_results = None
        st.rerun()
with col2:
    if st.button("Modèles Image", use_container_width=True,
                 type="primary" if st.session_state.comparison_mode == "image" else "secondary"):
        st.session_state.comparison_mode = "image"
        st.session_state.model_comparison_results = None
        st.rerun()

# Modèles disponibles
st.divider()
if st.session_state.comparison_mode == "text":
    st.header("Modèles Texte")
    models = get_available_text_models()
else:
    st.header("Modèles Image")
    models = get_available_image_models()

# Afficher les modèles
cols = st.columns(3)
for col, (model_id, config) in zip(cols, models.items()):
    with col:
        st.subheader(config.short_name)
        st.caption(config.description)
        st.metric("Confiance base", f"{config.base_confidence*100:.0f}%")

# Entrée
st.divider()
st.header("Test")

if st.session_state.comparison_mode == "text":
    designation = st.text_input("Désignation", placeholder="Ex: Console PlayStation 5")
    description = st.text_area("Description (optionnel)", height=80)

    if st.button("Comparer", type="primary", use_container_width=True):
        if not designation.strip():
            st.error("Veuillez saisir une désignation.")
        else:
            with st.spinner("Analyse..."):
                full_text = preprocess_product_text(designation, description)
                results = st.session_state.multi_model_classifier.predict_all_text_models(full_text)
                metrics = st.session_state.multi_model_classifier.get_comparison_metrics(results)
                st.session_state.model_comparison_results = {
                    "results": results, "metrics": metrics, "mode": "text"
                }
else:
    uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
    compare_image = None
    if uploaded_file:
        compare_image = load_image_from_upload(uploaded_file)
        is_valid, msg = validate_image(compare_image)
        if is_valid:
            st.image(compare_image, width=200)
        else:
            st.error(msg)
            compare_image = None

    if st.button("Comparer", type="primary", use_container_width=True):
        if compare_image is None:
            st.error("Veuillez uploader une image.")
        else:
            with st.spinner("Analyse..."):
                results = st.session_state.multi_model_classifier.predict_all_image_models(compare_image)
                metrics = st.session_state.multi_model_classifier.get_comparison_metrics(results)
                st.session_state.model_comparison_results = {
                    "results": results, "metrics": metrics, "mode": "image"
                }

# Résultats
if st.session_state.model_comparison_results:
    data = st.session_state.model_comparison_results
    results = data["results"]
    metrics = data["metrics"]
    mode = data["mode"]
    models_config = TEXT_MODELS if mode == "text" else IMAGE_MODELS

    st.divider()
    st.header("Résultats")

    # Métriques globales
    col1, col2, col3 = st.columns(3)
    col1.metric("Accord", f"{metrics['agreement_ratio']*100:.0f}%")
    col2.metric("Confiance moy.", f"{metrics['avg_confidence']*100:.1f}%")
    col3.metric("Meilleur", models_config[metrics["best_model"]].short_name)

    # Résultats par modèle
    st.divider()
    sorted_results = sorted(results.items(), key=lambda x: x[1].confidence, reverse=True)

    for rank, (model_id, result) in enumerate(sorted_results, 1):
        config = models_config[model_id]
        cat_name, _, cat_emoji = get_category_info(result.category)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.markdown(f"**#{rank}**")
        with col2:
            st.markdown(f"**{config.short_name}**: {cat_emoji} {cat_name}")
        with col3:
            st.markdown(f"**{result.confidence*100:.1f}%**")

    # Bar chart
    st.divider()
    bar_data = []
    for model_id, result in results.items():
        config = models_config[model_id]
        bar_data.append({"Modèle": config.short_name, "Confiance": result.confidence * 100})

    df = pd.DataFrame(bar_data)
    fig = px.bar(df, x="Confiance", y="Modèle", orientation='h',
                 color="Confiance", color_continuous_scale=['#FFE5E5', '#BF0000'])
    fig.update_layout(height=200, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("### Modèles")
    st.divider()
    st.markdown(f"**Mode**: {st.session_state.comparison_mode.upper()}")
    st.divider()
    st.markdown("**Texte**: SVM, RF, CamemBERT")
    st.markdown("**Image**: ResNet+SVM, ResNet+RF, VGG16")
    st.divider()
    if st.button("Réinitialiser"):
        st.session_state.model_comparison_results = None
        st.rerun()
