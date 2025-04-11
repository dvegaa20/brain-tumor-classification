import matplotlib.pyplot as plt


def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss over epochs.

    Parameters
    ----------
    history:
        The history object returned by the model's fit method.
    """
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, "b", label="Training acc")
    plt.plot(epochs, val_acc, "r", label="Validation acc")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()


# test_loss, test_acc = model.evaluate(test_generator)
# print(f"\nTest accuracy: {test_acc:.4f}")
