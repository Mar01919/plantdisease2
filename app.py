import streamlit as st
import tensorflow as tf
import numpy as np
import os
import tempfile

#-------------------------------
st.title("🧠 Reconocimiento de Enfermedades en Plantas")

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Elige Página", ["Inicio", "Acerca de", "Reconocimiento de enfermedad"])

# Variables globales
model = None
model_loaded = False

# Página de inicio
if app_mode == "Inicio":
    st.header("SISTEMA DE RECONOCIMIENTO DE ENFERMEDADES DE PLANTAS")
    image_path = os.path.join(os.path.dirname(__file__), "home_page.jpeg")
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    st.markdown("""
    ¡Bienvenido al Sistema de Reconocimiento de Enfermedades de las Plantas! 🌿🔍

    **Cómo funciona:**
    1. Ve a **Reconocimiento de enfermedad**.
    2. Sube tu modelo `.h5`.
    3. Luego sube una imagen de una hoja.
    4. El sistema te dará la predicción.

    **Ventajas:**
    - Precisión con deep learning.
    - Interfaz simple.
    - Resultados rápidos.
    """)

# Página "Acerca de"
elif app_mode == "Acerca de":
    st.header("Acerca del proyecto")
    st.markdown("""
    Este sistema detecta enfermedades en hojas de cultivos usando un modelo entrenado en TensorFlow. 
    El modelo puede reconocer más de 30 tipos de enfermedades y estados saludables.

    **Dataset:** Basado en PlantVillage (más de 80K imágenes).
    """)

# Página de predicción
elif app_mode == "Reconocimiento de enfermedad":
    st.header("🔬 Reconocimiento de enfermedad")

    # Subir modelo
    st.subheader("1️⃣ Sube tu modelo `.h5`")
    uploaded_model = st.file_uploader("Sube tu modelo entrenado (.h5)", type=["h5"])

    if uploaded_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            tmp_path = tmp.name
        try:
            model = tf.keras.models.load_model(tmp_path)
            model_loaded = True
            st.success("✅ Modelo cargado correctamente.")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
            model_loaded = False

    # Subir imagen si el modelo ya está cargado
    if model_loaded:
        st.subheader("2️⃣ Sube una imagen para predecir")
        test_image = st.file_uploader("Escoge una imagen de hoja", type=["jpg", "jpeg", "png"])

        if test_image is not None:
            st.image(test_image, caption="Imagen seleccionada", use_column_width=True)

            if st.button("🔍 Predecir"):
                try:
                    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
                    input_arr = tf.keras.preprocessing.image.img_to_array(image)
                    input_arr = np.expand_dims(input_arr, axis=0)

                    prediction = model.predict(input_arr)
                    result_index = np.argmax(prediction)

                    # Lista de clases (ajústala según tu modelo)
                    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                'Tomato___healthy']
                    
                    st.success(f"🌱 Predicción: **{class_name[result_index]}**")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error al procesar la imagen: {e}")
    else:
        st.info("⬆️ Sube un modelo primero para continuar.")
