import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Capítulo 8", page_icon="🎨", layout="wide")
st.title("🎨 Capítulo 8: Detector de Color Azul")

st.info("📸 La cámara se mostrará en tiempo real. Toma fotos para detectar objetos azules.")

# Controles de ajuste
with st.sidebar:
    st.header("⚙️ Ajustes de Detección")
    h_min = st.slider("Matiz Mínimo (H)", 0, 180, 100)
    h_max = st.slider("Matiz Máximo (H)", 0, 180, 130)
    s_min = st.slider("Saturación Mínima (S)", 0, 255, 100)
    v_min = st.slider("Valor Mínimo (V)", 0, 255, 100)
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Tip
    Ajusta los deslizadores si no detecta bien el azul
    """)

# Cámara en tiempo real - captura fotos
camera_photo = st.camera_input("Captura una foto")

if camera_photo is not None:
    # Convertir la foto capturada a formato OpenCV
    bytes_data = camera_photo.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Rangos personalizables para detectar azul en HSV
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, 255, 255])
    
    # Procesar la imagen
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = cv2.medianBlur(res, ksize=5)
    
    # Mostrar resultados en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Original")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("🎭 Máscara")
        st.image(mask, use_container_width=True, channels="GRAY")
    
    with col3:
        st.subheader("🔵 Detector")
        st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Información adicional
    pixels_detectados = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    porcentaje = (pixels_detectados / total_pixels) * 100
    
    st.metric("Píxeles azules detectados", f"{porcentaje:.2f}%")

else:
    st.warning("📸 Usa el botón de la cámara arriba para capturar una imagen")


