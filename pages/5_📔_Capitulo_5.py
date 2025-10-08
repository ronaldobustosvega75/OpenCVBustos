import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 5", page_icon="🔑", layout="wide")

st.title("🔑 Capítulo 5: Detección de Keypoints ORB")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    keypoints, descriptors = orb.compute(gray, keypoints)
    
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    img_rgb = cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB)
    
    st.metric("Keypoints detectados", len(keypoints))
    st.image(img_rgb, use_container_width=True)

else:
    st.info("👆 Sube una imagen para comenzar")