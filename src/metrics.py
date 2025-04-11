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
