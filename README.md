# 🏋️‍♂️ Potencia Predictiva 2.0: Modelo de Powerlifting Chileno

Este proyecto es una aplicación web interactiva que utiliza Machine Learning para predecir las marcas máximas (Sentadilla, Press de Banca y Peso Muerto) de un atleta de powerlifting basándose en sus características personales.

El objetivo es proporcionar una herramienta útil tanto para atletas nuevos que buscan metas realistas como para entrenadores que desean establecer benchmarks.

![Aquí puedes poner un pantallazo de tu app](https://powerlifting.cl/wp-content/uploads/2025/07/potencia-predictiva.png)

---

## 🛠️ Tecnologías Utilizadas

* **Python:** Lenguaje principal de programación.
* **Pandas:** Para la manipulación y limpieza de datos.
* **Scikit-learn:** Para la construcción del modelo de Machine Learning (Random Forest Regressor).
* **Streamlit:** Para la creación de la aplicación web interactiva.
* **Jupyter Notebook:** Para la exploración inicial de datos y prototipado del modelo.

---

## 🚀 Cómo Ejecutar el Proyecto

Para ejecutar esta aplicación en tu máquina local, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/albert-bq/analisis-powerlifters-chilenos.git](https://github.com/albert-bq/analisis-powerlifters-chilenos.git)
    cd analisis-powerlifters-chilenos
    ```

2.  **Crea un entorno virtual e instala las dependencias:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Ejecuta la aplicación de Streamlit:**
    ```bash
    streamlit run app.py
    ```

¡La aplicación se abrirá automáticamente en tu navegador!

---