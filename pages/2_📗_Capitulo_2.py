import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 2", page_icon="🔲", layout="wide")

st.title("🔲 Capítulo 2: Erosión y Dilatación")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    kernel = np.ones((5,5), np.uint8)
    
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("Erosión")
        st.image(img_erosion, use_container_width=True)
    
    with col3:
        st.subheader("Dilatación")
        st.image(img_dilation, use_container_width=True)

else:
    st.info("👆 Sube una imagen para comenzar")