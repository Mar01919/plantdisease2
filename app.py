import os
import gdown
#-------------------------------
import streamlit as st
import tensorflow as tf
import numpy as np
#-------------------------------
#DDDDDDDDDDDDDD
MODEL_PATH = "Eva1.h5"
MODEL_URL = "https://marcelacastillo.com/Eva1.h5"  # <-- cambia esto

if not os.path.exists(MODEL_PATH):
    try:
        st.info("Descargando modelo desde servidor HostGator...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
        else:
            st.error(f"❌ Error al descargar modelo. Código: {response.status_code}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Falló la descarga del modelo: {e}")
        st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("❌ Modelo no encontrado después de la descarga.")
    st.stop()
else:
    st.success("✅ Modelo descargado y verificado correctamente.")


#llllllllllllll

# Función de predicción
def model_prediction(test_image):
    model = tf.keras.models.load_model(MODEL_PATH)
    #model = tf.keras.models.load_model("trained_plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convertir en batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

#-------------------------------

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Elige Página",["Inicio","Acerca de","Reconocimiento de enfermedad"])

#Main Page
if(app_mode=="Inicio"):
    st.header("SISTEMA DE RECONOCIMIENTO DE ENFERMEDADES DE PLANTAS")
    #image_path = "home_page.jpeg"
    image_path = os.path.join(os.path.dirname(__file__), "home_page.jpeg")
    if not os.path.exists(image_path):
        st.error(f"No se encontró la imagen en la ruta: {os.path.abspath(image_path)}")
    else:
        st.image(image_path,use_column_width=True)
    #----- st.image(image_path,use_column_width=True)
    #st.image(image_path,use_container_width=True) 

    st.markdown("""
¡Bienvenido al Sistema de Reconocimiento de Enfermedades de las Plantas! 🌿🔍
    
Nuestra misión es ayudar a identificar enfermedades de las plantas de manera eficiente. Sube una imagen de una planta y nuestro sistema la analizará para detectar cualquier signo de enfermedades. ¡Juntos, protejamos nuestros cultivos y aseguremos una cosecha más saludable!

### Cómo funciona
    1. **Subir imagen:** Vaya a la página de **Reconocimiento de enfermedades** y cargue una imagen de una planta con sospechas de enfermedades.
    2. **Análisis:** Nuestro sistema procesará la imagen utilizando algoritmos avanzados para identificar posibles enfermedades.
    3. **Resultados:** Vea los resultados y las recomendaciones para futuras acciones.

### ¿Por qué elegirnos?
    - **Precisión:** Nuestro sistema utiliza técnicas de aprendizaje automático de última generación para la detección precisa de enfermedades.
    - **Fácil de usar:** Interfaz simple e intuitiva para una experiencia de usuario perfecta.
    - **Rápido y eficiente:** Reciba resultados en segundos, lo que permite una toma de decisiones rápida.

### Empezar
    Haga clic en la página de **Reconocimiento de Enfermedades** en la barra lateral para cargar una imagen y experimentar el poder de nuestro Sistema de Reconocimiento de Enfermedades de Plantas.

### Sobre Nosotros
    Obtenga más información sobre el proyecto, nuestro equipo y nuestros objetivos en la página **Acerca de**.
    """)

#About Project
elif(app_mode=="Acerca de"):
    st.header("Acerca de")
    st.markdown("""
                #### Acerca del conjunto de datos
                Este conjunto de datos se recrea mediante el aumento sin conexión del conjunto de datos original. El conjunto de datos original se puede encontrar en este repositorio de Github.
                Este conjunto de datos consta de aproximadamente 87K imágenes rgb de hojas de cultivos sanas y enfermas, que se clasifican en 38 clases diferentes. El conjunto de datos total se divide en una proporción de 80/20 de conjunto de entrenamiento y validación, conservando la estructura de directorios.
                Más adelante se crea un nuevo directorio que contiene 33 imágenes de prueba con fines de predicción.
                #### Contenido
                1. Entrenamiento (70295 imágenes)
                2. Prueba (33 imágenes)
                3. Validación (17572 imágenes)
                """)



#Prediction Page
elif(app_mode=="Reconocimiento de enfermedad"):
    st.header("Reconocimiento de enfermedad")
    test_image = st.file_uploader("Escoge una imagen:")
    if(st.button("Muestra Imagen")):
        st.image(test_image,width=4,use_column_width=True)
        #st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predicción")):
        st.balloons()
        #st.snow()
        st.write("Nuestra Predicción")
        result_index = model_prediction(test_image)
        #Reading Labels
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
        st.success("El modelo está prediciendo que es un {}".format(class_name[result_index]))