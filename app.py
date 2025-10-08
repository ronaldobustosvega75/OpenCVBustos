import streamlit as st
from utils import setup_page

st.set_page_config(page_title="Procesamiento de Imágenes", page_icon="🎨", layout="wide")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
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



