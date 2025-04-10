import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from preprocessing import load_and_preprocess_images

# Paths
base_dir = "data/raw"
augmented_dir = "data/augmented"
train_dir = os.path.join(base_dir, "training")

# Load processed images
train_images, train_labels, class_names = load_and_preprocess_images(train_dir)

if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Instance of ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
)

batch_size = 8
num_batches = 6

# Generator for augmented images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode="binary",
    save_to_dir=augmented_dir,
    save_prefix="aug",
    save_format="jpg",
)

# Display augmented images
for i in range(num_batches):
    images, labels = next(train_generator)

    plt.figure(figsize=(10, 5))
    for j in range(len(images)):
        plt.subplot(2, 4, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[j])
    plt.show()

print(f"\n{batch_size * num_batches} images have been augmented at: {augmented_dir}")
