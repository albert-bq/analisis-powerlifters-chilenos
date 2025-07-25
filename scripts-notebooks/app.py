import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Potencia Predictiva 2.0",
    page_icon="🏋️‍♂️",
    layout="centered"
)


# --- Cargar Modelo y Datos ---
# Cargar el nuevo pipeline multiobjetivo
try:
    model = joblib.load('powerlifting_multi_model.joblib')
except FileNotFoundError:
    st.error("Error: No se encontró el archivo del modelo 'powerlifting_multi_model.joblib'. Asegúrate de que esté en la misma carpeta que app.py.")
    st.stop()

# Cargar los datos originales para las opciones del menú
try:
    df_raw = pd.read_csv('powerliftingchile.csv')
    equipment_options = sorted(df_raw['Equipment'].dropna().unique())
except FileNotFoundError:
    equipment_options = ['Raw', 'Wraps', 'Single-ply', 'Multi-ply']


# --- Interfaz de Usuario ---
st.title("🏋️‍♂️ Potencia Predictiva 2.0")
st.write("Introduce tus datos para estimar tus marcas en Sentadilla, Press de Banca y Peso Muerto. Predicción basada en información de powerlifting.org y solo mide atletas chilenos")

# Columnas para los inputs
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox('Sexo', ('M', 'F'))
    age = st.number_input('Edad', min_value=13, max_value=80, value=25)

with col2:
    equipment = st.selectbox('Equipamiento', equipment_options)
    bodyweight = st.number_input('Peso Corporal (kg)', min_value=40.0, max_value=200.0, value=80.0, step=0.5)


# --- Predicción ---
if st.button('¡Predecir mis Levantamientos!', type="primary"):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'Age': [age],
        'BodyweightKg': [bodyweight],
        'Sex': [sex],
        'Equipment': [equipment]
    })

    # Realizar la predicción multiobjetivo
    prediction = model.predict(input_data)
    
    # Separar las predicciones
    squat_pred = prediction[0, 0]
    bench_pred = prediction[0, 1]
    deadlift_pred = prediction[0, 2]
    total_pred = squat_pred + bench_pred + deadlift_pred

    st.write("---")
    st.subheader("Tus Marcas Estimadas:")

    # Mostrar los resultados en 3 columnas
    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric(label="Sentadilla", value=f"{squat_pred:.1f} kg")
    with res2:
        st.metric(label="Press de Banca", value=f"{bench_pred:.1f} kg")
    with res3:
        st.metric(label="Peso Muerto", value=f"{deadlift_pred:.1f} kg")

    st.header(f"Total Estimado: {total_pred:.1f} kg")
    st.balloons()