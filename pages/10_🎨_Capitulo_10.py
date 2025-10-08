import streamlit as st
import cv2
import numpy as np
from pose_estimation import PoseEstimator

st.set_page_config(page_title="Capítulo 10", page_icon="🎯", layout="wide")
st.title("🎯 Capítulo 10: Realidad Aumentada - Pirámide 3D")

# Definir la función ANTES de usarla
def overlay_pyramid(img, tracked):
    """Dibuja una pirámide 3D sobre el objeto rastreado"""
    
    x_start, y_start, x_end, y_end = tracked.target.rect
    
    # Definir cuadrilátero 3D de referencia
    quad_3d = np.float32([
        [x_start, y_start, 0],
        [x_end, y_start, 0],
        [x_end, y_end, 0],
        [x_start, y_end, 0]
    ])
    
    # Parámetros de la cámara
    h, w = img.shape[:2]
    K = np.float64([
        [w, 0, 0.5*(w-1)],
        [0, w, 0.5*(h-1)],
        [0, 0, 1.0]
    ])
    dist_coef = np.zeros(4)
    
    # Resolver PnP para obtener rotación y traslación
    ret, rvec, tvec = cv2.solvePnP(quad_3d, tracked.quad, K, dist_coef)
    
    # Vértices de la pirámide (base + vértice superior)
    pyramid_height = 4  # Altura de la pirámide
    overlay_vertices = np.float32([
        [0, 0, 0],           # Base: esquina inferior izquierda
        [0, 1, 0],           # Base: esquina superior izquierda
        [1, 1, 0],           # Base: esquina superior derecha
        [1, 0, 0],           # Base: esquina inferior derecha
        [0.5, 0.5, pyramid_height]  # Vértice superior (centro)
    ])
    
    # Escalar y trasladar vértices
    scale = [(x_end-x_start), (y_end-y_start), -(x_end-x_start)*0.3]
    verts = overlay_vertices * scale + (x_start, y_start, 0)
    
    # Proyectar vértices 3D a 2D
    verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
    verts = np.int32(verts)
    
    # Dibujar base de la pirámide
    cv2.drawContours(img, [verts[:4]], -1, (0, 255, 0), -3)
    
    # Dibujar caras laterales con diferentes colores
    colors = [
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 0, 150),    # Rojo oscuro
        (255, 255, 0)   # Cian
    ]
    
    for i in range(4):
        face = np.vstack([
            verts[i:i+1],
            verts[(i+1)%4:(i+1)%4+1],
            verts[4:5]
        ])
        cv2.drawContours(img, [face], -1, colors[i], -3)
    
    # Dibujar aristas de la pirámide
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
    for i, j in edges:
        pt1 = tuple(verts[i])
        pt2 = tuple(verts[j])
        cv2.line(img, pt1, pt2, (0, 0, 0), 2)


st.info("📸 Captura una foto y luego selecciona el área del objeto a rastrear haciendo clic y arrastrando")

# Inicializar session_state
if 'tracker' not in st.session_state:
    st.session_state.tracker = PoseEstimator()
    st.session_state.target_image = None
    st.session_state.target_rect = None
    st.session_state.selection_mode = True

# Sidebar con controles
with st.sidebar:
    st.header("⚙️ Controles")
    
    if st.button("🗑️ Limpiar objetivos", use_container_width=True):
        st.session_state.tracker = PoseEstimator()
        st.session_state.target_rect = None
        st.success("Objetivos limpiados")
    
# Captura de imagen objetivo
camera_photo = st.camera_input("📸 Captura imagen")

if camera_photo is not None:
    # Convertir foto a formato OpenCV
    bytes_data = camera_photo.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    h, w = frame.shape[:2]
    
    # Modo de selección: Definir rectángulo objetivo
    if st.session_state.selection_mode:
        st.subheader("🎯 Paso 1: Define el área del objeto a rastrear")
        
        col1, col2 = st.columns(2)
        with col1:
            x_start = st.slider("X inicial", 0, w-1, w//4)
            y_start = st.slider("Y inicial", 0, h-1, h//4)
        
        with col2:
            x_end = st.slider("X final", 0, w-1, 3*w//4)
            y_end = st.slider("Y final", 0, h-1, 3*h//4)
        
        # Dibujar rectángulo en la imagen
        img_with_rect = frame.copy()
        cv2.rectangle(img_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        st.image(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB), 
                use_container_width=True,
                caption="Ajusta el rectángulo verde sobre el objeto")
        
        if st.button("✅ Confirmar selección y activar rastreo", type="primary", use_container_width=True):
            rect = (x_start, y_start, x_end, y_end)
            st.session_state.tracker.add_target(frame, rect)
            st.session_state.target_image = frame.copy()
            st.session_state.target_rect = rect
            st.session_state.selection_mode = False
            st.success("✅ Objetivo registrado! Ahora captura nuevas fotos para ver la pirámide 3D")
            st.rerun()
    
    # Modo de rastreo: Mostrar pirámide 3D
    else:
        st.subheader("🔍 Rastreando objeto...")
        
        # Procesar frame para rastreo
        tracked = st.session_state.tracker.track_target(frame)
        
        img_result = frame.copy()
        
        if tracked:
            for item in tracked:
                # Dibujar contorno del objeto rastreado
                cv2.polylines(img_result, [np.int32(item.quad)], True, (0, 0, 0), 2)
                
                # Dibujar puntos de características
                for (x, y) in np.int32(item.points_cur):
                    cv2.circle(img_result, (x, y), 2, (0, 0, 0))
                
                # Dibujar pirámide 3D
                overlay_pyramid(img_result, item)
            
            status = "✅ Objeto detectado"
            status_color = "green"
        else:
            status = "❌ Objeto no detectado - Intenta con mejor ángulo o iluminación"
            status_color = "red"
        
        # Mostrar resultado
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(st.session_state.target_image, cv2.COLOR_BGR2RGB),
                    caption="Imagen objetivo original",
                    use_container_width=True)
        
        with col2:
            st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB),
                    caption="Rastreo actual con pirámide 3D",
                    use_container_width=True)
        
        st.markdown(f"**Estado:** :{status_color}[{status}]")
        
        if st.button("🔄 Seleccionar nuevo objetivo"):
            st.session_state.selection_mode = True
            st.rerun()

else:
    st.warning("📸 Captura una imagen para comenzar")
    
