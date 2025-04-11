from keras import optimizers
from utils.augmentation import create_generators
from utils.models import (
    build_basic_cnn,
    build_enhanced_cnn,
    build_vgg16,
    build_mobilenet,
)
from keras.callbacks import EarlyStopping
from utils.visualization import plot_training_history

EPOCHS = 10

# Data Generators
train_gen, val_gen, test_gen = create_generators(
    train_dir="data/raw/training",
    test_dir="data/raw/testing",
    use_rgb=True,
)

# Models
models = {
    "basic_cnn": build_basic_cnn,
    "enhanced_cnn": build_enhanced_cnn,
    "vgg16": build_vgg16,
    "mobilenet": build_mobilenet,
}
selected_model = "vgg16"
model = models[selected_model]()
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_acc", mode="max", patience=5, restore_best_weights=True),
]

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    metrics=["acc"],
)

# Train the model
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
)

# Save the model
model.save("models/best_model_build_vgg16.keras")
print("Model saved!")


# Evaluación final
plot_training_history(history)
