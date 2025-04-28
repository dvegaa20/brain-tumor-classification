from keras import optimizers
from utils.augmentation import create_generators
from utils.models import (
    build_basic_cnn,
    build_enhanced_cnn,
    build_vgg16,
    build_mobilenet,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.visualization import plot_training_history

EPOCHS = 50

# Data Generators
train_gen, val_gen, _ = create_generators(
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
selected_model = "mobilenet"
model = models[selected_model]()
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
]

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=1e-5),
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
model.save("models/best_model_mobilenet.keras")
print("Model saved!")


# Evaluación final
plot_training_history(history)
