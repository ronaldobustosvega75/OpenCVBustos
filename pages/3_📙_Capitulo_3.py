import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 3", page_icon="🖱️", layout="wide")

st.title("🖱️ Capítulo 3: Inversión Interactiva")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.write("**Instrucciones:** Usa los controles para definir una región a invertir")
    
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("Posición X", 0, img.shape[1]-100, 50)
        y = st.slider("Posición Y", 0, img.shape[0]-100, 50)
    with col2:
        w = st.slider("Ancho", 50, img.shape[1]-x, 100)
        h = st.slider("Alto", 50, img.shape[0]-y, 100)
    
    img_output = img_rgb.copy()
    img_output[y:y+h, x:x+w] = 255 - img_output[y:y+h, x:x+w]
    cv2.rectangle(img_output, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Original")
        st.image(img_rgb, use_container_width=True)
    with col_b:
        st.subheader("Con Inversión")
        st.image(img_output, use_container_width=True)

else:
    st.info("👆 Sube una imagen para comenzar")