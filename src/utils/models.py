from keras import layers, models
from keras.applications import MobileNetV2, VGG16

IMG_SIZE_GRAY = (150, 150, 1)
IMG_SIZE_RGB = (150, 150, 3)


def build_basic_cnn(input_shape=IMG_SIZE_GRAY, num_classes=4):
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


def build_enhanced_cnn(input_shape=IMG_SIZE_GRAY, num_classes=4):
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(10, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


def build_vgg16(input_shape=IMG_SIZE_RGB, num_classes=4):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = models.Sequential()

    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def build_mobilenet(input_shape=IMG_SIZE_RGB, num_classes=4):
    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )

    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = models.Sequential()

    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
