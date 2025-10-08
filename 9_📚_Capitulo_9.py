import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 9", page_icon="🔍", layout="wide")

st.title("🔍 Capítulo 9: Detectores Dense y SIFT")

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Dense Detector
    step_size = 20
    feature_scale = 20
    img_bound = 5
    keypoints_dense = []
    rows, cols = img.shape[:2]
    for x in range(img_bound, rows, feature_scale):
        for y in range(img_bound, cols, feature_scale):
            keypoints_dense.append(cv2.KeyPoint(float(x), float(y), step_size))
    
    img_dense = cv2.drawKeypoints(img.copy(), keypoints_dense, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # SIFT Detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_sift = sift.detect(gray, None)
    img_sift = cv2.drawKeypoints(img.copy(), keypoints_sift, None, 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Dense Detector ({len(keypoints_dense)} pts)")
        st.image(cv2.cvtColor(img_dense, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader(f"SIFT Detector ({len(keypoints_sift)} pts)")
        st.image(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    st.info("👆 Sube una imagen para comenzar")