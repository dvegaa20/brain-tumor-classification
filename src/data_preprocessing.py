import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

IMG_SIZE = 128


def load_and_preprocess_images(data_path):
    """
    Load and preprocess images from a specified directory.

    This function reads images from a directory where each subdirectory
    corresponds to a different class. Images are converted to grayscale,
    resized to a fixed size, and normalized to a range of [0, 1]. The
    images and their corresponding labels are shuffled randomly.

    Args:
        data_path (str): Path to the directory containing image data,
                         with subdirectories representing class labels.

    Returns:
        tuple: A tuple containing:
            - images (np.ndarray): Array of preprocessed images.
            - labels (np.ndarray): Array of integer labels corresponding
                                   to image classes.
            - class_names (list): Sorted list of class names.
    """

    images = []
    labels = []
    class_names = os.listdir(data_path)
    class_names.sort()

    class_counts = {
        cls: len(os.listdir(os.path.join(data_path, cls))) for cls in class_names
    }
    print(f"Distribución de imágenes en {data_path}: {class_counts}")

    # Cargar y preprocesar las imágenes
    for label, cls in enumerate(class_names):
        class_path = os.path.join(data_path, cls)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if os.path.isfile(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)

    # Convertir a arrays de NumPy
    images = np.array(images)
    labels = np.array(labels)

    # Normalizar las imágenes
    images = images / 255.0

    # Randomizar el orden de las imágenes y etiquetas
    images, labels = shuffle(images, labels, random_state=42)
    return images, labels, class_names


# Cargar datasets de entrenamiento y prueba
train_data_path = "data/raw/training"
test_data_path = "data/raw/testing"

train_images, train_labels, class_names = load_and_preprocess_images(train_data_path)
test_images, test_labels, _ = load_and_preprocess_images(test_data_path)

# Primera imagen (entrenamiento)
plt.figure()
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# Grid de imágenes (entrenamiento)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"{train_labels[i]} - {class_names[train_labels[i]]}")
plt.show()
