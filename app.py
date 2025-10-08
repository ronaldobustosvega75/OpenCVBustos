import streamlit as st

st.set_page_config(page_title="Procesamiento de Imágenes", page_icon="🎨", layout="wide")

hide_streamlit_style = """
    <style>
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .viewerBadge_container__1QSob {display: none;}
    
    /* Mantener visible el menú de 3 puntos */
    #MainMenu {visibility: visible !important;}
    #stHeader {visibility: visible !important;}
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



















