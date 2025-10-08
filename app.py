import streamlit as st
from utils import setup_page

st.set_page_config(page_title="Procesamiento de Imágenes", page_icon="🎨", layout="wide")

hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    button[kind="header"] {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    [data-testid="stDecoration"] {display: none;}
    
    /* Ocultar solo el contenido del header (Fork, GitHub) pero mantener el header visible */
    header[data-testid="stHeader"] {visibility: visible !important;}
    header[data-testid="stHeader"] > div:first-child {visibility: hidden;}
    
    /* Asegurar que las flechitas de navegación estén visibles */
    [data-testid="stHeader"] button[kind="header"] {display: block !important; visibility: visible !important;}
    
    /* Mantener visible el menú de 3 puntos */
    #MainMenu {visibility: visible !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🎨 Procesamiento Digital de Imágenes")
st.markdown("""
### Bienvenido
Explora el procesamiento de imágenes con OpenCV.
**Navega usando el menú lateral ⬅️**
""")
st.info("👈 Selecciona un capítulo del menú lateral para comenzar")



