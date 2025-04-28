from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from utils.preprocessing import get_data_info, plot_sample_images

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42


def create_generators(train_dir, test_dir, use_rgb):
    """
    Creates three generators for training, validation and testing using the given directories.

    Parameters
    ----------
    train_dir : str
        Path to the training directory.
    test_dir : str
        Path to the testing directory.
    use_rgb : bool
        Flag to determine if RGB or grayscale images should be used.

    Returns
    -------
    train_generator :
        Generator for training data.
    val_generator :
        Generator for validation data.
    test_generator :
        Generator for testing data.
    """
    if use_rgb:
        color_mode = "rgb"
    else:
        color_mode = "grayscale"

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        validation_split=0.2,
    )

    test_val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = None
    val_generator = None
    test_generator = None

    if train_dir:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=color_mode,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=SEED,
        )

        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=color_mode,
            class_mode="categorical",
            subset="validation",
            shuffle=True,
            seed=SEED,
        )

    if test_dir:
        test_generator = test_val_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=color_mode,
            class_mode="categorical",
            shuffle=False,
        )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    train_dir = "data/raw/training"
    test_dir = "data/raw/testing"

    class_names = get_data_info(train_dir)

    train_gen, val_gen, test_gen = create_generators(train_dir, test_dir)

    print("")
    print("Sample images from training generator:")
    plot_sample_images(train_gen, class_names)

    print("Sample images from validation generator:")
    plot_sample_images(val_gen, class_names)

    print("Sample images from testing generator:")
    plot_sample_images(test_gen, class_names)
