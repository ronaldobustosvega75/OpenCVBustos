import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 4", page_icon="👤", layout="wide")

st.title("👤 Capítulo 4: Detección de Rostros")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rostros detectados", len(face_rects))
    with col2:
        st.metric("Confianza", "Haar Cascade")
    
    st.image(img_rgb, use_container_width=True)

else:
    st.info("👆 Sube una imagen para comenzar")