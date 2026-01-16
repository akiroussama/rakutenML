"""
Configuration centralisée pour l'application Streamlit Rakuten.

Ce module définit tous les chemins, paramètres et constantes utilisés
dans l'application de classification de produits.
"""
from pathlib import Path

# =============================================================================
# Chemins du projet
# =============================================================================
# Racine du projet (remonte de 3 niveaux depuis ce fichier)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Chemins des données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = RAW_DATA_DIR / "images"

# Chemins des modèles
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_MODEL_PATH = MODELS_DIR / "image_classifier.joblib"
TEXT_MODEL_PATH = MODELS_DIR / "text_classifier.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
RESNET_EXTRACTOR_PATH = MODELS_DIR / "resnet50_extractor.h5"
CATEGORY_MAPPING_PATH = MODELS_DIR / "category_mapping.json"

# Chemins des features pré-extraites
IMPLEMENTATION_DIR = PROJECT_ROOT / "implementation"
FEATURES_DIR = IMPLEMENTATION_DIR / "outputs"
METADATA_PATH = FEATURES_DIR / "metadata_augmented.json"

# Chemins des assets Streamlit
STREAMLIT_DIR = Path(__file__).parent
ASSETS_DIR = STREAMLIT_DIR / "assets"
EXAMPLES_DIR = ASSETS_DIR / "examples"

# =============================================================================
# Configuration de l'application
# =============================================================================
APP_CONFIG = {
    "title": "Rakuten Product Classifier",
    "icon": "🛒",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# =============================================================================
# Configuration des modèles
# =============================================================================
MODEL_CONFIG = {
    # Mode mock: utilise des prédictions simulées (pour dev/test)
    "use_mock": True,

    # Poids pour la fusion multimodale (image, texte)
    "fusion_weights": (0.6, 0.4),

    # Nombre de top prédictions à afficher
    "top_k": 5,

    # Seuil de confiance minimum pour affichage
    "confidence_threshold": 0.1,
}

# =============================================================================
# Configuration des images
# =============================================================================
IMAGE_CONFIG = {
    # Taille cible pour le preprocessing
    "target_size": (224, 224),

    # Formats acceptés
    "allowed_formats": ["jpg", "jpeg", "png", "webp"],

    # Taille max en MB
    "max_size_mb": 10,
}

# =============================================================================
# Configuration du texte
# =============================================================================
TEXT_CONFIG = {
    # Longueur max du texte (caractères)
    "max_length": 5000,

    # Langues supportées pour la détection
    "supported_languages": ["fr", "en", "de", "it", "es", "pt"],
}

# =============================================================================
# Couleurs et thème Rakuten
# =============================================================================
THEME = {
    "primary_color": "#BF0000",  # Rouge Rakuten
    "secondary_color": "#FFFFFF",
    "background_color": "#F5F5F5",
    "text_color": "#333333",
    "success_color": "#28A745",
    "warning_color": "#FFC107",
    "error_color": "#DC3545",
}
