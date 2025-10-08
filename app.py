import streamlit as st

st.set_page_config(page_title="Procesamiento de Imágenes", page_icon="🎨", layout="wide")

hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    button[kind="header"] {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    [data-testid="stDecoration"] {display: none;}

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











