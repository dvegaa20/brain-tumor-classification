from keras import layers, models
from keras.applications import MobileNetV2, VGG16

IMG_SIZE = (150, 150, 1)


def build_basic_cnn(input_shape=IMG_SIZE, num_classes=4):
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


def build_enhanced_cnn(input_shape=IMG_SIZE, num_classes=4):
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(10, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


# def enhanced_cnn(input_shape=(150, 150, 3), num_classes=4):
#     model = models.Sequential()
#     model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=input_shape))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(128, activation="relu"))
#     model.add(layers.Dense(num_classes, activation="sigmoid"))
#     return model


def build_vgg16(input_shape=IMG_SIZE, num_classes=4):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential()

    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


# def transfer_vgg16(input_shape=(150, 150, 3), num_classes=4):
#     base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
#     base_model.trainable = False
#     model = models.Sequential(
#         [
#             base_model,
#             layers.GlobalAveragePooling2D(),
#             layers.Dense(128, activation="relu"),
#             layers.Dense(num_classes, activation="sigmoid"),
#         ]
#     )
#     return model


def build_mobilenet(input_shape=IMG_SIZE, num_classes=4):
    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential()

    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


# def transfer_mobilenet(input_shape=(150, 150, 3), num_classes=4):
#     base_model = MobileNetV2(
#         input_shape=input_shape, include_top=False, weights="imagenet"
#     )
#     base_model.trainable = False
#     model = models.Sequential(
#         [
#             base_model,
#             layers.GlobalAveragePooling2D(),
#             layers.Dense(128, activation="relu"),
#             layers.Dense(num_classes, activation="sigmoid"),
#         ]
#     )
#     return model
