import streamlit as st
import cv2
import numpy as np

def setup_page(title, icon="🎨", layout="wide", hide_header=True):
    """
    Configura la página con estilos comunes y oculta elementos de Streamlit
    
    Args:
        title: Título de la página
        icon: Emoji del ícono
        layout: Layout de la página (wide o centered)
        hide_header: Si True, oculta GitHub/Fork pero mantiene los 3 puntitos
    """
    st.set_page_config(page_title=title, page_icon=icon, layout=layout)
    
    if hide_header:
        # Ocultar GitHub, Fork y Deploy, pero mantener los 3 puntitos
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
            [data-testid="stToolbar"] {display: none;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
