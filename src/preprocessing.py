import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def get_data_info(data_path):
    """
    Prints the number of images in each class in the given directory.

    Args:
        data_path (str): The path to the directory containing the image data.

    Returns:
        A list of class names.
    """
    class_names = sorted(os.listdir(data_path))
    class_counts = {
        cls: len(os.listdir(os.path.join(data_path, cls))) for cls in class_names
    }

    print(f"\nImage distribution in {data_path}:")
    for cls, count in class_counts.items():
        print(f"- {cls}: {count} images")

    return class_names


def plot_sample_images(generator, class_names, num_samples=25):
    """
    Plots a specified number of sample images from a data generator.

    Args:
        generator: A generator of (images, labels) tuples.
        class_names: A list of class names in the order they are expected in the labels.
        num_samples (int): The number of sample images to plot (default is 25).
    """
    images, labels = next(generator)

    plt.figure(figsize=(10, 10))
    for i in range((min(num_samples, len(images)))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].squeeze(), cmap=plt.cm.binary)
        plt.xlabel(f"{labels[i]} - {class_names[int(labels[i])]}")
    plt.tight_layout()
    plt.show()
