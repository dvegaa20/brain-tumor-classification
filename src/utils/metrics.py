from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_classification_report(true_labels, predictions, class_names):
    report = classification_report(
        true_labels,
        np.argmax(predictions, axis=1),
        target_names=class_names,
    )
    print("\nClassification Report:\n", report)
    return report


def plot_confusion_matrix(true_labels, predictions, class_names):
    cm = confusion_matrix(
        true_labels,
        np.argmax(predictions, axis=1),
        labels=range(len(class_names)),
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()
