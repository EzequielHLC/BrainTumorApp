import tensorflow as tf

# Carga del Modelo
model = tf.keras.models.load_model("NEURAID_v2.h5")

# Generar la imagen
# Show_shapes=True para ver como cambian los datos
tf.keras.utils.plot_model(
    model,
    to_file="modelo_tumores_v2.png",
    show_shapes=True,
    show_layer_names=True,
    show_layer_activations=True
)