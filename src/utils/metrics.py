from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_classification_report(true_labels, predictions, class_names):
    """
    Generates a classification report based on the given true labels and predictions.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the data.
    predictions : array-like
        The predictions of the model.
    class_names : list-like
        A list of class names in the order they are expected in the labels.

    Returns
    -------
    report : str
        The classification report as a string.
    """
    
    report = classification_report(
        true_labels,
        np.argmax(predictions, axis=1),
        target_names=class_names,
    )
    print("\nClassification Report:\n", report)
    return report


def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Plots a confusion matrix using the given true labels and predictions.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the data.
    predictions : array-like
        The predictions of the model.
    class_names : list-like
        A list of class names in the order they are expected in the labels.

    Notes
    -----
    The plot is a heatmap using seaborn's heatmap function with the following settings:
    - x-axis labels are the predicted labels
    - y-axis labels are the true labels
    - the color bar is on the right
    - the title is "Confusion Matrix"
    - the x-axis label is "Predicted label"
    - the y-axis label is "True label"
    """
    
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
