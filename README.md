# Brain Tumor Classification Project
> Proyecto de clasificación de imágenes con preprocesamiento, aumento de datos y entrenamiento de modelo en TensorFlow.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Este proyecto implementa una pipeline completa de clasificación de imágenes, que incluye:

- **Preprocesamiento de datos**: Carga, normalización y visualización de imágenes.
- **Aumento de datos**: Generación de imágenes aumentadas para mejorar la generalización del modelo.
- **Entrenamiento del modelo**: Red neuronal convolucional en TensorFlow/Keras.

![](header.png)

## Instalación

### OS X & Linux:
```sh
pip install -r requirements.txt
```

### Windows:
```sh
pip install -r requirements.txt
```

## Uso

### 1. Preprocesamiento de datos
Ejecuta el script para cargar, normalizar y visualizar imágenes:
```sh
python data_preprocessing.py
```

### 2. Aumento de datos
Genera imágenes aumentadas a partir del dataset:
```sh
python data_augmentation.py
```

### 3. Entrenamiento del modelo
Entrena la red neuronal con las imágenes preprocesadas y aumentadas:
```sh
python train_model.py
```

### 4. Evaluación
Evalúa el modelo con imágenes de prueba:
```sh
python evaluate_model.py
```

## Estructura del proyecto
```
📂 image-classification-project
├── data
│   ├── raw
│   │   ├── training
│   │   ├── testing
│   ├── augmented
│
├── notebooks
│   ├── exploratory_analysis.ipynb
│
├── src
│   ├── data_preprocessing.py
│   ├── data_augmentation.py
│   ├── train_model.py
│   ├── evaluate_model.py
│
├── models
│   ├── model.h5
│
├── requirements.txt
├── README.md
```

## Configuración para desarrollo
Para configurar el entorno de desarrollo, instala las dependencias y verifica que TensorFlow está correctamente instalado:
```sh
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Historial de versiones

* 1.0.0
    * Primera versión estable con preprocesamiento, aumento de datos y entrenamiento de modelo.

## Meta

Diego Vega – [@your_twitter](https://twitter.com/your_twitter) – your_email@example.com

Distribuido bajo la licencia MIT. Ver ``LICENSE`` para más información.

[https://github.com/diegovega/image-classification-project](https://github.com/diegovega/image-classification-project)

## Contribuciones

1. Haz un fork (<https://github.com/diegovega/image-classification-project/fork>)
2. Crea una rama para tu feature (`git checkout -b feature/nueva_funcionalidad`)
3. Realiza los cambios y haz commit (`git commit -am 'Agregada nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva_funcionalidad`)
5. Crea un nuevo Pull Request

