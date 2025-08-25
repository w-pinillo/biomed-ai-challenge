# Clasificación Multi-etiqueta de Literatura Médica

## Descripción

Este proyecto implementa una solución de machine learning para clasificar automáticamente artículos de investigación médica en una o más de las siguientes categorías: `Cardiovascular`, `Neurológica`, `Hepatorrenal` y `Oncológica`. Se utiliza un enfoque de modelo híbrido para maximizar el rendimiento.

## Características

- **Modelo Híbrido**: Combina un modelo clásico (Regresión Logística o XGBoost) con predicciones de un modelo Transformer (BioBERT).
- **Ingeniería de Características**: Utiliza embeddings de texto generados por BioBERT junto con características de dominio específicas (conteo de palabras clave médicas) para enriquecer los datos de entrada.
- **Ensamble**: Emplea un promedio ponderado de las predicciones de los modelos para mejorar la precisión y robustez del resultado final.
- **Reproducibilidad**: Incluye un archivo `requirements.txt` para una configuración rápida y consistente del entorno de desarrollo.

## Resultados

El modelo de ensamble final, combinando una Regresión Logística con el modelo base de BioBERT, alcanzó un **F1-score ponderado de 0.87** en el conjunto de datos de prueba.

## Análisis de Errores

Para una comprensión más profunda del rendimiento del modelo, se realiza un análisis de errores detallado. Este análisis incluye:

-   **Matrices de Confusión por Etiqueta**: Se generan matrices de confusión binarias para cada una de las etiquetas (Cardiovascular, Neurológica, Hepatorrenal, Oncológica), lo que permite visualizar los verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos.
-   **Casos de Ejemplo**: Se proporcionan ejemplos de textos que corresponden a verdaderos positivos, falsos positivos y falsos negativos para cada etiqueta. Esto es crucial para identificar patrones de error y áreas de mejora del modelo.

Este análisis se ejecuta automáticamente como parte del script de evaluación del ensamble (`src/models/ensemble.py`).

## Estructura del Repositorio

```
├── data/
│   ├── medical_articles.csv      # Datos brutos
│   ├── preprocessed_articles.csv # Datos procesados
│   └── biobert_embeddings.npy    # Embeddings generados
├── models/
│   ├── classical_*.joblib        # Modelos clásicos entrenados
│   └── ...
├── notebooks/
│   └── analysis.ipynb            # Análisis Exploratorio de Datos (EDA)
├── src/
│   ├── preprocess.py             # Script de preprocesamiento
│   └── models/
│       ├── classical.py          # Script para modelos clásicos
│       └── ensemble.py           # Script para el ensamble
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este archivo
```

## Instalación

Sigue estos pasos para configurar el entorno de desarrollo localmente.

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/w-pinillo/biomed-ai-challenge.git
   cd biomed-ai-challenge
   ```

2. **Crear un entorno virtual**
   Este comando crea un entorno virtual en una carpeta llamada `venv`.
   ```bash
   python3 -m venv venv
   ```

3. **Activar el entorno virtual**

   #### En macOS y Linux (bash/zsh)
   ```bash
   source venv/bin/activate
   ```

   #### En Windows (Command Prompt / PowerShell)
   ```bash
   venv\Scripts\activate
   ```

4. **Instalar las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar el pipeline completo de principio a fin, tienes dos opciones:

1.  **Entrenar y Evaluar (Proceso Completo):**
    Este es el método recomendado para una ejecución completa que incluye el preprocesamiento de datos, el entrenamiento del modelo clásico y la evaluación del ensamble.

    ```bash
    bash scripts/run_e2e.sh
    ```

2.  **Solo Evaluar (Usar Modelos Pre-entrenados):**
    Si ya has entrenado los modelos o has descargado los artefactos pre-entrenados, puedes saltarte el paso de entrenamiento del modelo clásico. Esto es útil para una evaluación rápida o para verificar los resultados sin consumir recursos de entrenamiento.

    ```bash
    bash scripts/run_e2e.sh --skip-training
    ```

Pasos para realizar inferencia con un archivo CSV:

1. **Evaluar la Solución con un Archivo CSV**
   Este script permite cargar un archivo CSV, realizar predicciones y, si el CSV incluye las etiquetas verdaderas, evaluar el rendimiento del modelo.

   ```bash
   python3 -m src.evaluate_solution --input_csv data/medical_articles.csv
   ```
   Reemplaza `data/medical_articles.csv` con la ruta a tu archivo CSV de entrada.

## Estrategia del Modelo

El repositorio incluye un modelo pre-entrenado (classical_logistic_regression_tuned.joblib, 46 KB) para permitir ejecutar las predicciones directamente sin reentrenar.

Se utiliza un enfoque híbrido para aprovechar las fortalezas de diferentes tipos de modelos.

-   **BioBERT como Extractor de Características**: BioBERT, un modelo Transformer pre-entrenado en literatura biomédica, se utiliza como un potente **extractor de características** que comprende la semántica del dominio. Se decidió no realizar un fine-tuning completo del modelo BioBERT debido al tamaño limitado del dataset (3,565 observaciones) y al alto costo computacional y tiempo de entrenamiento (aproximadamente 10 horas). La experiencia sugiere que, para datasets de este tamaño, el fine-tuning podría no ofrecer una mejora significativa que justifique la inversión, siendo más eficiente y práctico utilizar sus embeddings pre-entrenados.

-   **Modelos Clásicos (Regresión Logística)**: Este modelo es eficaz para aprender patrones a partir de los embeddings de BioBERT y de las características de dominio diseñadas a mano. Se comparó con XGBoost, y la Regresión Logística fue seleccionada por su rendimiento similar y menor complejidad/tiempo de entrenamiento.

-   **Ensamble**: Finalmente, el **ensamble** combina las señales predictivas del modelo clásico (Regresión Logística) y del modelo BioBERT base. Esta combinación permite aprovechar las fortalezas complementarias de ambos enfoques, logrando un rendimiento final más alto y robusto (F1-score ponderado de 0.87) que cualquiera de los modelos por separado.
