import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 1", page_icon="🎨", layout="wide")

st.title("🎨 Capítulo 1: Fusión de Canales de Color")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Separar y combinar canales
    b, g, r = cv2.split(img)
    
    gbr_img = cv2.merge((g, b, r))
    rbr_img = cv2.merge((r, b, r))
    
    # Convertir a RGB para mostrar
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gbr_rgb = cv2.cvtColor(gbr_img, cv2.COLOR_BGR2RGB)
    rbr_rgb = cv2.cvtColor(rbr_img, cv2.COLOR_BGR2RGB)
    
    # Mostrar resultados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original")
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.subheader("GBR")
        st.image(gbr_rgb, use_container_width=True)
    
    with col3:
        st.subheader("RBR")
        st.image(rbr_rgb, use_container_width=True)

else:
    st.info("👆 Sube una imagen para comenzar")