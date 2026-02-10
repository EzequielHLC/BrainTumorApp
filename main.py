import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import io
import base64

# --- Configuración de la API ---
app = FastAPI(
    title="API de Clasificación de Tumores Cerebrales",
    description="Una API que utiliza un modelo de Deep Learning para clasificar MRI de tumores cerebrales."
)

# Configurar CORS (Cross-Origin Resource Sharing)
# Esto permite que tu frontend (que estará en otro 'origen') llame a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (POST, GET, etc.)
    allow_headers=["*"],
)

# Configurar archivos estáticos (favicon, CSS, JS, etc.)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# --- MODELO Y FUNCIONES (Tu código, ligeramente adaptado) ---

# @st.cache_resource se reemplaza por @app.on_event("startup") o simplemente cargándolo globalmente
# Por simplicidad, lo cargaremos una vez al inicio.
# (Para producción real, se usaría un manejo de 'lifespan' o 'startup event')

print("Cargando modelos, esto puede tardar un momento...")

MODELO_GLOBAL, GRAD_MODEL_GLOBAL = None, None

def cargar_modelo():
    """
    Cargo el modelo de IA desde un archivo .h5.
    """
    global MODELO_GLOBAL, GRAD_MODEL_GLOBAL
    
    if MODELO_GLOBAL is None:
        model = load_model("NEURAID_v2.h5")
        
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

        grad_model = Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )
        MODELO_GLOBAL = model
        GRAD_MODEL_GLOBAL = grad_model
        print("Modelos cargados exitosamente.")
    
    return MODELO_GLOBAL, GRAD_MODEL_GLOBAL

# Llamamos a la función una vez al inicio para cargar los modelos en memoria
cargar_modelo()


def preprocesar_imagen(image_bytes, target_size=(224, 224)):
    """
    Preproceso la imagen (en bytes) para que coincida con la entrada del modelo.
    """
    # Abre la imagen desde los bytes en memoria
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array_normalized = img_array / 255.0  # Normalización
    img_array_expanded = np.expand_dims(img_array_normalized, axis=0)

    # Devolvemos el array expandido (para predicción) y el array original (para Grad-CAM)
    return img_array_expanded, img_array

def generar_grad_cam(grad_model, img_array_expanded, class_index):
    """
    Genera el mapa de calor de Grad-CAM (tu código sin cambios)
    """
    img_tensor = tf.convert_to_tensor(img_array_expanded, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if tf.equal(max_val, 0):
        return np.zeros_like(heatmap.numpy())
    heatmap = heatmap / max_val
    return heatmap.numpy()

def superponer_heatmap(original_img_array, heatmap, alpha=0.5): # Aumenté alpha
    """
    Superpone el mapa de calor sobre la imagen original (tu código sin cambios)
    """
    heatmap_resized = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    original_img_uint8 = np.uint8(original_img_array)
    superimposed_img = cv2.addWeighted(original_img_uint8, 1 - alpha, heatmap_color, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img_rgb

def convertir_imagen_a_base64(img_array):
    """
    Convierte un array de NumPy (imagen) a una cadena Base64 
    para poder enviarla como JSON.
    """
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64

# --- ENDPOINT DE LA API ---

@app.get("/favicon.ico")
async def favicon():
    """
    Endpoint para servir el favicon.
    Retorna una respuesta 204 No Content para evitar errores 404.
    """
    return Response(status_code=204)

@app.get("/")
async def root():
    """
    Endpoint para servir el index.html en la raíz.
    """
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>API de Clasificación de Tumores Cerebrales</h1><p>Sube una imagen en /predict</p>")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint de predicción. Recibe un archivo de imagen,
    realiza la predicción y devuelve los resultados, incluyendo imágenes Base64.
    """
    # 1. Leer y preprocesar la imagen
    image_bytes = await file.read()
    processed_img_expanded, original_img_array = preprocesar_imagen(image_bytes)
    
    # 2. Obtener modelos
    modelo, grad_model = cargar_modelo()

    # 3. Realizar la predicción
    predictions = modelo.predict(processed_img_expanded)
    
    # 4. Obtener resultados
    class_names = ['Glioma', 'Meningioma', 'Sin Tumor', 'Tumor Pituitario']
    score_index = int(np.argmax(predictions)) # Convertir a int nativo de Python
    predicted_class = class_names[score_index]
    confidence = float(predictions[0][score_index] * 100) # Convertir a float nativo
    
    all_scores = {class_names[i]: f"{predictions[0][i] * 100:.2f}%" for i in range(len(class_names))}

    # 5. Generar Grad-CAM
    heatmap = generar_grad_cam(grad_model, processed_img_expanded, score_index)
    superimposed_image_array = superponer_heatmap(original_img_array, heatmap)

    # 6. Convertir imágenes a Base64 para enviarlas por JSON
    # El frontend (JavaScript) decodificará esto
    original_b64 = convertir_imagen_a_base64(original_img_array)
    grad_cam_b64 = convertir_imagen_a_base64(superimposed_image_array)

    # 7. Devolver la respuesta como JSON
    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "all_scores": all_scores,
        "images": {
            "original_base64": original_b64,
            "grad_cam_base64": grad_cam_b64
        }
    }

# --- Punto de entrada para ejecutar la app ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)