import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import numpy as np
import cv2
import os
import time

# --- Configuración de la página ---

st.set_page_config(
    page_title="Clasificador de Tumores Cerebrales",
    page_icon="🧠",
    layout="centered"
)

# --- Funciones Auxiliares ---

@st.cache_resource
def cargar_modelo():
    """
    Cargo el modelo de IA desde un archivo .h5.
    """
    model = load_model("NEURAID_v2.h5")

    # Intento obtener la capa 'out_relu' si existe; si no, busco la última capa con salida 4D (conv)
    try:
        conv_layer = model.get_layer('out_relu')
    except Exception:
        conv_layer = None
        for layer in reversed(model.layers):
            try:
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    conv_layer = layer
                    break
            except Exception:
                continue
        if conv_layer is None:
            raise ValueError("No se encontró una capa convolucional válida para Grad-CAM.")

    # Usar model.input (singular) para evitar mismatch en la estructura de inputs
    grad_model = Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )
    return model, grad_model

def preprocesar_imagen(image, target_size=(224, 224)):
    """
    Preproceso la imagen subida por el usuario para que coincida con la entrada del modelo (224x224, normalizada).
    """
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array_normalized = img_array / 255.0  # Normalización
    img_array_expanded = np.expand_dims(img_array_normalized, axis=0)

    # Devolvemos 3 versiones:
    # 1. El array normalizado (para Grad-CAM)
    # 2. El array expandido (para predicción)
    # 3. El array original (para mostrar)

    return img_array_normalized, img_array_expanded, np.array(img.resize(target_size))


def analizar_imagen(image, caption_text=None):
    """
    Realiza preprocesado, predicción y genera Grad-CAM para una imagen PIL.
    """
    if caption_text:
        st.image(image, caption=caption_text, use_container_width=True)
    else:
        st.image(image, caption='Imagen seleccionada.', use_container_width=True)

    # Barra de progreso: creamos un único placeholder y lo actualizamos
    progress_ph = st.empty()
    update_progress(progress_ph, 1)

    processed_img_norm, processed_img_expanded, original_img_array = preprocesar_imagen(image)
    update_progress(progress_ph, 2)

    with st.spinner('Clasificando...'):
        predictions = modelo.predict(processed_img_expanded)
    update_progress(progress_ph, 3)

    class_names = ['Glioma', 'Meningioma', 'Sin Tumor', 'Tumor Pituitario']

    score_index = np.argmax(predictions)
    predicted_class = class_names[score_index]
    confidence = predictions[0][score_index] * 100

    st.success(f"**Predicción:** {predicted_class}")
    st.info(f"**Confianza:** {confidence:.2f}%")

    st.write("---")
    st.subheader("Explicabilidad del Modelo (Grad-CAM)")
    st.write("El mapa de calor resalta las áreas que el modelo consideró más importantes para su predicción.")

    with st.spinner('Generando mapa de calor...'):
        heatmap = generar_grad_cam(grad_model, processed_img_expanded, score_index)
        superimposed_image = superponer_heatmap(original_img_array, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img_array, caption='Imagen Original (Procesada)', use_container_width=True)
        with col2:
            st.image(superimposed_image, caption='Imagen con Grad-CAM', use_container_width=True)

    st.write("Probabilidades detalladas:")
    all_scores = {class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
    st.dataframe(all_scores, use_container_width=True)

    # Marcar resultado final (y mantener una sola aparición de la barra)
    update_progress(progress_ph, 4)


def update_progress(placeholder, current_step: int):
    """
    Actualiza (en un único placeholder) la barra de progreso estética.
    """
    steps = ['Preprocesado', 'Predicción', 'Grad-CAM', 'Resultado']
    total = len(steps)
    if total <= 1:
        percent = 100
    else:
        percent = int(((current_step - 1) / (total - 1)) * 100)

    html = f"""
    <style>
    .prog-container {{background:#f0f2f6;border-radius:12px;padding:10px;margin-bottom:14px;}}
    .prog-bar-outer {{background:#e6e9ee;border-radius:10px;padding:4px;}}
    .prog-bar-inner {{width:{percent}%;height:16px;background:linear-gradient(90deg,#4facfe,#00f2fe);border-radius:8px;transition:width:600ms ease;}}
    .step-labels {{display:flex;justify-content:space-between;margin-top:8px;font-size:13px;color:#333;font-weight:600;}}
    .step {{flex:1;text-align:center}}
    .dot {{display:inline-block;width:14px;height:14px;border-radius:50%;background:#ddd;margin-bottom:6px}}
    .dot.completed {{background:#4caf50}}
    .dot.active {{background:linear-gradient(90deg,#4facfe,#00f2fe);box-shadow:0 2px 6px rgba(0,0,0,0.12)}}
    </style>
    <div class="prog-container">
      <div class="prog-bar-outer"><div class="prog-bar-inner"></div></div>
      <div class="step-labels">
    """

    for i, label in enumerate(steps, start=1):
        cls = ''
        if i < current_step:
            cls = 'completed'
        elif i == current_step:
            cls = 'active'
        html += f"<div class='step'><div class='dot {cls}'></div><div>{label}</div></div>"

    html += "</div></div>"
    placeholder.markdown(html, unsafe_allow_html=True)
    # Pequeña pausa para que la animación sea visible
    time.sleep(0.5)

def generar_grad_cam(grad_model, img_array_expanded, class_index):
    """
    Genera el mapa de calor de Grad-CAM
    """
    # Aseguro tensor float32 y batch dimension correcta
    img_tensor = tf.convert_to_tensor(img_array_expanded, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Obtengo las salidas del modelo Grad-CAM
        conv_outputs, predictions = grad_model(img_tensor)

        # Si predictions viene como lista/tuple, tomo el primer elemento
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # Quiero la pérdida de la clase específica
        loss = predictions[:, class_index]

    # Obtengo los gradientes de la pérdida respecto a las salidas conv
    grads = tape.gradient(loss, conv_outputs)

    # Pool de gradientes y ponderación de mapas de características
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # eliminar dimensión batch

    # Combino mapas de características y gradientes (suma ponderada)
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Normalización (ReLU y escalado seguro)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if tf.equal(max_val, 0):
        return np.zeros_like(heatmap.numpy())
    heatmap = heatmap / max_val

    return heatmap.numpy()

def superponer_heatmap(original_img_array, heatmap, alpha=0.4):
    """
    Superpone el mapa de calor sobre la imagen original.
    """
    # Convierto el heatmap a 8-bit (0-255)
    heatmap_resized = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Me asuguro de que la imagen original esté en formato uint8
    original_img_uint8 = np.uint8(original_img_array)

    # Superpongo el heatmap sobre la imagen original
    superimposed_img = cv2.addWeighted(original_img_uint8, 1 - alpha, heatmap_color, alpha, 0)

    # Convierto de BGR (OpenCV) a RGB (Streamlit)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img_rgb


# --- Interfaz Principal de la Aplicación ---

# Título de la aplicación
st.title("🧠 Clasificador de Tumores Cerebrales por MRI")
st.write("""Sube una imagen de Resonancia Magnética (MRI) cerebral y 
el modelo hará una predicción de si se trata de un glioma, meningioma, 
un tumor pituitario, o si no hay tumor.
""")

# Paso 1: Cargar el modelo
with st.spinner('Cargando el modelo de IA, por favor espera...'):
    modelo, grad_model = cargar_modelo()
st.success('Modelo cargado exitosamente!', icon="✅")

# Paso 2: Widget para subir la imagen
uploaded_file = st.file_uploader(
    "Elige una imagen de Resonancia Magnética (MRI) cerebral...",
    type=["jpg", "jpeg", "png"]
)

# Opción: usar imágenes de prueba desde la carpeta `test_img`
use_test_images = st.checkbox('Usar imágenes desde el directorio de Imágenes de Prueba')

if use_test_images:
    test_root = os.path.join(os.getcwd(), 'test_img')
    if not os.path.exists(test_root):
        st.error('No se encontró la carpeta test_img en el directorio de trabajo.')
    else:
        classes = [d for d in sorted(os.listdir(test_root)) if os.path.isdir(os.path.join(test_root, d))]
        if not classes:
            st.error('La carpeta test_img no contiene subcarpetas.')
        else:
            selected_class = st.selectbox('Selecciona la carpeta/clase', classes)
            class_dir = os.path.join(test_root, selected_class)
            images = [f for f in sorted(os.listdir(class_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                st.warning('No hay imágenes en la carpeta seleccionada.')
            else:
                selected_image = st.selectbox('Selecciona la imagen', images)
                image_path = os.path.join(class_dir, selected_image)
                try:
                    image = Image.open(image_path)
                    analizar_imagen(image, caption_text=f'Imagen de prueba: {selected_class}/{selected_image}')
                except Exception as e:
                    st.error(f'No se pudo abrir la imagen: {e}')
else:
    # Flujo original: subir archivo
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            analizar_imagen(image, caption_text='Imagen subida.')
        except Exception as e:
            st.error(f'Error al leer la imagen subida: {e}')