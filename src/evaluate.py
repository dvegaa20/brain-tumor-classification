import tensorflow as tf
from utils.augmentation import create_generators
from utils.metrics import generate_classification_report, plot_confusion_matrix


def evaluate_model(model_path, test_dir):
    """
    Evaluate a model on a test set.

    Parameters
    ----------
    model_path : str
        Path to the model to evaluate.
    test_dir : str
        Path to the test set directory.

    Notes
    -----
    This function prints out the test loss and accuracy, a classification
    report and a confusion matrix, using the test set.
    """
    model = tf.keras.models.load_model(model_path)

    _, _, test_gen = create_generators(train_dir=None, test_dir=test_dir, use_rgb=True)

    results = model.evaluate(test_gen)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")

    predictions = model.predict(test_gen)
    true_labels = test_gen.classes

    generate_classification_report(true_labels, predictions, test_gen.class_indices)
    plot_confusion_matrix(true_labels, predictions, test_gen.class_indices)


if __name__ == "__main__":
    model_path = "models/best_model_build_vgg16.keras"
    test_dir = "data/raw/testing"

    evaluate_model(model_path, test_dir)
