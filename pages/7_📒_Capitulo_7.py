import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 7", page_icon="📐", layout="wide")

st.title("📐 Capítulo 7: Detección y Suavizado de Contornos")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    factor = st.slider("Factor de suavizado", 0.001, 0.1, 0.01, 0.001)
    
    smoothen_contours = []
    for contour in contours:
        epsilon = factor * cv2.arcLength(contour, True)
        smoothen_contours.append(cv2.approxPolyDP(contour, epsilon, True))
    
    contour_img = img.copy()
    cv2.drawContours(contour_img, smoothen_contours, -1, color=(0, 0, 255), thickness=3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader(f"Contornos Suavizados ({len(contours)})")
        st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    st.info("👆 Sube una imagen para comenzar")