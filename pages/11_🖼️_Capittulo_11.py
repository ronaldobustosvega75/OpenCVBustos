import streamlit as st
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from pathlib import Path
import tempfile
from utils import setup_page

st.set_page_config(page_title="Capítulo 11", page_icon="🔍", layout="wide")

class DenseDetector():
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound

    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints

class FeatureExtractor(object):
    def __init__(self, num_clusters=64):
        self.sift = cv2.SIFT_create()
        self.num_clusters = num_clusters

    def extract_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = DenseDetector().detect(img)
        kps, des = self.sift.compute(gray, kps)
        return np.array(des, dtype=np.float32)

    def build_codebook(self, images):
        all_features = []
        for img_path, _ in images[:80]:
            img = cv2.imread(img_path)
            img = self.resize_image(img)
            features = self.extract_features(img)
            all_features.extend(features)
        
        all_features = np.array(all_features, dtype=np.float32)
        kmeans = KMeans(self.num_clusters, n_init=15, max_iter=100, random_state=42)
        kmeans.fit(all_features)
        return kmeans

    def get_feature_vector(self, img, kmeans):
        features = self.extract_features(img)
        labels = kmeans.predict(features)
        histogram = np.zeros(self.num_clusters)
        for label in labels:
            histogram[label] += 1
        return (histogram / np.sum(histogram)).reshape(1, -1).astype(np.float32)

    def resize_image(self, img, size=150):
        h, w = img.shape[:2]
        scale = size / min(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))

class Classifier():
    def __init__(self, input_size, num_classes):
        self.ann = cv2.ml.ANN_MLP_create()
        hidden_size = int((input_size * 2/3) + num_classes)
        self.ann.setLayerSizes(np.array([input_size, hidden_size, num_classes], dtype=np.int32))
        self.ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
        self.ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 300, 0.00001))
        self.ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.0001, 0.1)
        self.le = preprocessing.LabelBinarizer()
        self.labels = None

    def train(self, X, y):
        self.labels = np.unique(y)
        self.le.fit(self.labels)
        y_encoded = self.le.transform(y).astype(np.float32)
        if y_encoded.shape[1] == 1:
            y_encoded = np.hstack([1 - y_encoded, y_encoded])
        self.ann.train(X, cv2.ml.ROW_SAMPLE, y_encoded)

    def predict(self, X):
        _, probs = self.ann.predict(X)
        gato_prob = probs[0][0] * 100
        perro_prob = probs[0][1] * 100
        class_idx = np.argmax(probs[0])
        confidence = np.max(probs[0]) * 100
        return self.labels[class_idx], confidence, gato_prob, perro_prob

def load_dataset(gatos_path, perros_path):
    dataset = []
    for path in os.listdir(gatos_path):
        if path.lower().endswith(('.jpg', '.jpeg', '.png')):
            dataset.append((os.path.join(gatos_path, path), 'gato'))
    for path in os.listdir(perros_path):
        if path.lower().endswith(('.jpg', '.jpeg', '.png')):
            dataset.append((os.path.join(perros_path, path), 'perro'))
    return dataset

def augment_data(img):
    augmented = [img]
    augmented.append(cv2.flip(img, 1))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 10, 1.0)
    augmented.append(cv2.warpAffine(img, M, (w, h)))
    M = cv2.getRotationMatrix2D((w/2, h/2), -10, 1.0)
    augmented.append(cv2.warpAffine(img, M, (w, h)))
    return augmented

@st.cache_resource
def train_model():
    gatos_path = "pages/gatos"
    perros_path = "pages/perros"
    
    if not os.path.exists(gatos_path) or not os.path.exists(perros_path):
        return None, None, None
    
    dataset = load_dataset(gatos_path, perros_path)
    if len(dataset) == 0:
        return None, None, None
    
    np.random.shuffle(dataset)
    
    with st.spinner("🔄 Construyendo codebook con más iteraciones..."):
        extractor = FeatureExtractor(num_clusters=64)
        kmeans = extractor.build_codebook(dataset)
    
    with st.spinner("🔄 Extrayendo características de todas las imágenes..."):
        X, y = [], []
        for img_path, label in dataset:
            img = cv2.imread(img_path)
            img = extractor.resize_image(img)
            augmented_imgs = augment_data(img)
            for aug_img in augmented_imgs[:2]:
                fv = extractor.get_feature_vector(aug_img, kmeans)
                X.append(fv[0])
                y.append(label)
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
    
    with st.spinner("🔄 Entrenando red neuronal (esto puede tardar un poco)..."):
        classifier = Classifier(X.shape[1], 2)
        classifier.train(X, y)
    
    return classifier, kmeans, extractor

st.title("🔍 Capítulo 11: Clasificador de Gatos y Perros")

st.info("🎓 **Entrenamiento:** Este modelo está entrenado con **400+ muestras** aumentadas de 200 imágenes originales")

gatos_path = Path("pages/gatos")
perros_path = Path("pages/perros")
num_gatos = len(list(gatos_path.glob("*.jpg"))) + len(list(gatos_path.glob("*.png"))) if gatos_path.exists() else 0
num_perros = len(list(perros_path.glob("*.jpg"))) + len(list(perros_path.glob("*.png"))) if perros_path.exists() else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🐱 Gatos", num_gatos)
col2.metric("🐶 Perros", num_perros)
col3.metric("📊 Total", num_gatos + num_perros)

with col4:
    if st.button("🔄 Reentrenar", type="secondary", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

st.divider()

classifier, kmeans, extractor = train_model()

if classifier is None:
    st.error("❌ No se encontraron imágenes en 'pages/gatos' y 'pages/perros'")
    st.stop()

st.success("✅ Modelo entrenado correctamente")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Imagen Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("🎯 Análisis del Modelo")
        
        img_resized = extractor.resize_image(image)
        fv = extractor.get_feature_vector(img_resized, kmeans)
        label, confidence, gato_prob, perro_prob = classifier.predict(fv)
        
        # Debug: verificar valores
        # st.write(f"Debug - label: {label}, conf: {confidence:.2f}, gato: {gato_prob:.2f}, perro: {perro_prob:.2f}")
        
        # Mostrar probabilidades de ambas clases
        st.markdown("### 📊 Probabilidades:")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("🐱 Gato", f"{gato_prob:.1f}%")
            st.progress(float(gato_prob / 100))
        
        with col_b:
            st.metric("🐶 Perro", f"{perro_prob:.1f}%")
            st.progress(float(perro_prob / 100))
        
        st.divider()
        
        # Decisión final
        diferencia = abs(gato_prob - perro_prob)
        
        if confidence < 55:
            st.error("⚠️ NO ES GATO NI PERRO")
            st.warning(f"Confianza muy baja: {confidence:.1f}%")
        elif diferencia < 15:
            st.warning("🤔 NO ESTOY SEGURO")
            st.info(f"Las probabilidades son muy similares (diferencia: {diferencia:.1f}%)")
            st.caption(f"Predicción tentativa: {label.upper()}")
        else:
            if label == "gato":
                st.success("🐱 **ES UN GATO**")
            else:
                st.success("🐶 **ES UN PERRO**")
            st.metric("✅ Confianza", f"{confidence:.1f}%")
        
        # Información adicional
        with st.expander("ℹ️ Ver detalles técnicos"):
            st.write(f"**Vector de características:** {fv.shape}")
            st.write(f"**Predicción:** {label}")
            st.write(f"**Diferencia entre clases:** {diferencia:.2f}%")
            if diferencia < 15:

                st.warning("⚠️ Considera reentrenar con más imágenes o imágenes de mejor calidad")
