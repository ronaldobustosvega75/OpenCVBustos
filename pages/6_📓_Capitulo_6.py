import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 6", page_icon="✂️", layout="wide")

st.title("✂️ Capítulo 6: Seam Carving (Remoción de Objetos)")

st.info("⚠️ Este capítulo requiere interacción avanzada. Versión simplificada disponible.")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    energy = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("Mapa de Energía")
        st.image(energy, use_container_width=True)

else:
    st.info("👆 Sube una imagen para ver el mapa de energía")