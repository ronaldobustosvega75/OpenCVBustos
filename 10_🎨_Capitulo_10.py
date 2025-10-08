import streamlit as st
import cv2
import numpy as np
from pose_estimation import PoseEstimator, ROISelector

st.set_page_config(page_title="Capítulo 10", page_icon="🎯", layout="wide")
st.title("🎯 Capítulo 10: Realidad Aumentada")

if 'tracker' not in st.session_state:
    st.session_state.tracker = None
    st.session_state.rect = None
    st.session_state.paused = False

run = st.checkbox("Activar cámara")
col1, col2 = st.columns([3, 1])

with col2:
    if st.button("Limpiar objetivos"):
        if st.session_state.tracker:
            st.session_state.tracker.clear_targets()
    paused = st.checkbox("Pausar", value=st.session_state.paused)
    st.session_state.paused = paused

with col1:
    FRAME_WINDOW = st.empty()

class StreamlitTracker:
    def __init__(self, scaling_factor=0.8):
        self.cap = cv2.VideoCapture(0)
        self.rect = None
        self.scaling_factor = scaling_factor
        self.tracker = PoseEstimator()
        
        self.overlay_vertices = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0.5, 0.5, 4]])
        self.overlay_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0,4), (1,4), (2,4), (3,4)]
        self.color_base = (0, 255, 0)
        self.color_lines = (0, 0, 0)
        self.graphics_counter = 0
        self.time_counter = 0

    def set_target(self, frame, rect):
        self.rect = rect
        self.tracker.add_target(frame, rect)

    def clear_targets(self):
        self.tracker = PoseEstimator()
        self.rect = None

    def process_frame(self, paused=False):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.resize(frame, None, fx=self.scaling_factor, fy=self.scaling_factor, 
                          interpolation=cv2.INTER_AREA)
        img = frame.copy()
        
        if not paused and self.rect:
            tracked = self.tracker.track_target(frame)
            for item in tracked:
                cv2.polylines(img, [np.int32(item.quad)], True, self.color_lines, 2)
                for (x, y) in np.int32(item.points_cur):
                    cv2.circle(img, (x, y), 2, self.color_lines)
                self.overlay_graphics(img, item)
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def overlay_graphics(self, img, tracked):
        x_start, y_start, x_end, y_end = tracked.target.rect
        quad_3d = np.float32([[x_start, y_start, 0], [x_end, y_start, 0],
                              [x_end, y_end, 0], [x_start, y_end, 0]])
        h, w = img.shape[:2]
        K = np.float64([[w, 0, 0.5*(w-1)], [0, w, 0.5*(h-1)], [0, 0, 1.0]])
        dist_coef = np.zeros(4)
        ret, rvec, tvec = cv2.solvePnP(quad_3d, tracked.quad, K, dist_coef)
        
        self.time_counter += 1
        if not self.time_counter % 20:
            self.graphics_counter = (self.graphics_counter + 1) % 8
        
        self.overlay_vertices = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                                           [0.5, 0.5, self.graphics_counter]])
        
        verts = self.overlay_vertices * [(x_end-x_start), (y_end-y_start), 
                                         -(x_end-x_start)*0.3] + (x_start, y_start, 0)
        verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
        verts_floor = np.int32(verts).reshape(-1,2)
        
        cv2.drawContours(img, [verts_floor[:4]], -1, self.color_base, -3)
        cv2.drawContours(img, [np.vstack((verts_floor[:2], verts_floor[4:5]))], -1, (0,255,0), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[1:3], verts_floor[4:5]))], -1, (255,0,0), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[2:4], verts_floor[4:5]))], -1, (0,0,150), -3)
        cv2.drawContours(img, [np.vstack((verts_floor[3:4], verts_floor[0:1], verts_floor[4:5]))], -1, (255,255,0), -3)
        
        for i, j in self.overlay_edges:
            (x_start, y_start), (x_end, y_end) = verts[i], verts[j]
            cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), self.color_lines, 2)

    def release(self):
        self.cap.release()

if run:
    if st.session_state.tracker is None:
        st.session_state.tracker = StreamlitTracker()
    
    frame = st.session_state.tracker.process_frame(st.session_state.paused)
    if frame is not None:
        FRAME_WINDOW.image(frame, use_container_width=True)
    else:
        st.error("No se pudo acceder a la cámara")
else:
    if st.session_state.tracker:
        st.session_state.tracker.release()
        st.session_state.tracker = None
    st.warning("Activa la cámara para comenzar")