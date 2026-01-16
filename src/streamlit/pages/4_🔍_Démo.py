"""
Page de classification de produits Rakuten.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.category_mapping import get_category_info
from utils.mock_classifier import DemoClassifier, TEXT_MODELS, IMAGE_MODELS
from utils.image_utils import load_image_from_upload, validate_image
from utils.preprocessing import preprocess_product_text
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Démo - {APP_CONFIG['title']}",
    page_icon="🔍",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Session state
if "classifier" not in st.session_state:
    st.session_state.classifier = DemoClassifier()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Exemples
EXAMPLES = [
    ("Livre", "Harry Potter à l'école des sorciers", "Roman fantastique J.K. Rowling"),
    ("Console", "Console PlayStation 5", "Jeux vidéo Sony nouvelle génération"),
    ("Piscine", "Piscine gonflable ronde", "Piscine été jardin 3m diamètre"),
    ("Figurine", "Figurine Pop Marvel Spider-Man", "Funko Pop vinyle collection"),
    ("Téléphone", "iPhone 15 Pro Max 256GB", "Smartphone Apple OLED 5G"),
    ("Jouet", "LEGO Star Wars Millennium Falcon", "Jeu construction 1351 pièces"),
]

# Header
st.title("Classification de Produits")
st.markdown("Identifiez la catégorie Rakuten de vos produits.")

if st.session_state.get("use_mock", True):
    st.warning("Mode Démonstration")

st.divider()

# Tabs
tab_text, tab_image, tab_examples = st.tabs(["Texte", "Image", "Exemples"])

with tab_text:
    st.subheader("Classification par Texte")

    designation = st.text_input("Désignation", placeholder="Ex: Console PlayStation 5")
    description = st.text_area("Description (optionnel)", height=80)

    if st.button("Classifier", key="btn_text", type="primary", use_container_width=True):
        if not designation.strip():
            st.error("Veuillez saisir une désignation.")
        else:
            with st.spinner("Classification..."):
                text = preprocess_product_text(designation, description)
                result = st.session_state.classifier.predict(text=text, top_k=5)
                st.session_state.last_result = result
                st.session_state.last_image = None

with tab_image:
    st.subheader("Classification par Image")

    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        image = load_image_from_upload(uploaded)
        is_valid, msg = validate_image(image)

        if is_valid:
            st.image(image, width=200)
            if st.button("Classifier", key="btn_image", type="primary", use_container_width=True):
                with st.spinner("Classification..."):
                    result = st.session_state.classifier.predict(image=image, top_k=5)
                    st.session_state.last_result = result
                    st.session_state.last_image = image
        else:
            st.error(msg)

with tab_examples:
    st.subheader("Exemples")

    cols = st.columns(3)
    for i, (name, designation, description) in enumerate(EXAMPLES):
        with cols[i % 3]:
            if st.button(name, key=f"ex_{i}", use_container_width=True):
                with st.spinner("Classification..."):
                    text = preprocess_product_text(designation, description)
                    result = st.session_state.classifier.predict(text=text, top_k=5)
                    st.session_state.last_result = result
                    st.session_state.last_image = None
                    st.session_state.last_example = (name, designation)

# Résultats
if st.session_state.last_result:
    result = st.session_state.last_result
    cat_name, cat_full, cat_emoji = get_category_info(result.category)
    conf = result.confidence * 100

    st.divider()
    st.header("Résultat")

    # Catégorie principale
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"## {cat_emoji} {cat_name}")
        st.caption(cat_full)
        st.markdown(f"Code: `{result.category}`")

    with col2:
        color = "#28A745" if conf >= 70 else "#FF9800" if conf >= 40 else "#DC3545"
        st.metric("Confiance", f"{conf:.1f}%")
        st.progress(conf / 100)

    # Image si disponible
    if st.session_state.get("last_image") is not None:
        st.image(st.session_state.last_image, width=150, caption="Image analysée")

    # Top 5
    st.divider()
    st.subheader("Top 5 Prédictions")

    top5_data = []
    for code, score in result.top_k_predictions[:5]:
        name, _, emoji = get_category_info(code)
        top5_data.append({"Catégorie": f"{emoji} {name}", "Confiance": score * 100})

    df = pd.DataFrame(top5_data)
    fig = px.bar(df, x="Confiance", y="Catégorie", orientation='h',
                 color="Confiance", color_continuous_scale=['#FFE5E5', '#BF0000'])
    fig.update_layout(height=250, showlegend=False, coloraxis_showscale=False,
                      yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("### Classification")
    st.divider()

    st.markdown("**Modèles**")
    st.markdown("- Texte: CamemBERT")
    st.markdown("- Image: ResNet50+SVM")

    st.divider()
    if st.button("Réinitialiser"):
        st.session_state.last_result = None
        st.session_state.last_image = None
        st.rerun()
