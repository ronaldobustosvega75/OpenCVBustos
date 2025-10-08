import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="Capítulo 8", page_icon="🎨", layout="wide")
st.title("🎨 Capítulo 8: Detector de Color Azul en Tiempo Real")

# Configuración para WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ColorDetector(VideoTransformerBase):
    """Detector de color azul en tiempo real"""
    
    def __init__(self):
        self.lower_blue = np.array([100, 100, 100])
        self.upper_blue = np.array([130, 255, 255])
        self.show_mask = False
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convertir a HSV
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para detectar azul
        mask = cv2.inRange(hsv_frame, self.lower_blue, self.upper_blue)
        
        # Aplicar la máscara
        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.medianBlur(res, ksize=5)
        
        # Mostrar máscara o resultado
        if self.show_mask:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            return res

# Controles
st.sidebar.header("⚙️ Controles")
show_original = st.sidebar.checkbox("Mostrar video original", value=False)
show_mask = st.sidebar.checkbox("Mostrar máscara binaria", value=False)

# Contenedor para el video
col1, col2 = st.columns(2)

if show_original:
    with col1:
        st.subheader("📹 Video Original")
        webrtc_streamer(
            key="original",
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False}
        )
    
    with col2:
        st.subheader("🔵 Detector de Azul")
        ctx = webrtc_streamer(
            key="detector",
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=ColorDetector,
            media_stream_constraints={"video": True, "audio": False}
        )
        
        if ctx.video_transformer:
            ctx.video_transformer.show_mask = show_mask
else:
    st.subheader("🔵 Detector de Color Azul")
    ctx = webrtc_streamer(
        key="detector_full",
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=ColorDetector,
        media_stream_constraints={"video": True, "audio": False}
    )
    
    if ctx.video_transformer:
        ctx.video_transformer.show_mask = show_mask
