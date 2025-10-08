import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 8", page_icon="🎨", layout="wide")
st.title("🎨 Capítulo 8: Detector de Color Azul")

run = st.checkbox("Activar cámara")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    FRAME_ORIGINAL = st.empty()
with col2:
    st.subheader("Detector")
    FRAME_DETECTOR = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    lower = np.array([60, 100, 100])
    upper = np.array([180, 255, 255])
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara")
            break
        
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        res = cv2.medianBlur(res, ksize=5)
        
        FRAME_ORIGINAL.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        FRAME_DETECTOR.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    
    cap.release()
else:
    st.warning("Activa la cámara para comenzar")