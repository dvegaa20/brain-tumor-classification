import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image
from tkinter import filedialog, Label

# model1 = load_model("./models/best_model_basic_cnn.keras")
# model2 = load_model("./models/best_model_enhanced_cnn.keras")
model1 = load_model("./models/best_model_build_vgg16.keras")
model2 = load_model("./models/best_model_mobilenet.keras")

class_names = [
    "glioma",
    "meningioma",
    "no tumor",
    "pituitary",
]


def predict_image(model, img_path, target_size=(150, 150), grayscale=False):
    """
    Predicts the class of one image using a given model.

    Parameters
    ----------
    model:
        The model to use for prediction.
    img_path:
        The path to the image to predict.
    target_size: (150, 150)
        The size of the image to resize to.
    grayscale: False
        Whether to load the image in grayscale or not.

    Returns
    -------
        The predicted class of the image.
    """

    img = image.load_img(
        img_path,
        target_size=target_size,
        color_mode="grayscale" if grayscale else "rgb",
    )
    img_array = image.img_to_array(img)
    if grayscale:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)
    return pred_class[0]


def open_images():
    """
    Opens a file dialog for selecting multiple images, displays each image, and predicts
    their classes using two different models, displaying the results.

    Notes
    -----
    This function updates the GUI to show each selected image and appends the prediction
    results from two models to a text label.

    Uses
    ----
    - `filedialog.askopenfilenames()` to select images.
    - `Image.open()` to load images.
    - `ImageTk.PhotoImage()` for image display in the GUI.
    - `predict_one_image()` to predict image classes.

    Modifies
    --------
    - Updates `img_label` to display the currently selected image.
    - Updates `result_label` to display prediction results.
    """

    img_paths = filedialog.askopenfilenames()
    if img_paths:
        results_text = ""
        canvas.delete("all")
        image_refs.clear()
        image_size = 120
        overlap = 60
        total_width = (len(img_paths) - 1) * overlap + image_size
        x_offset = max((500 - total_width) // 2, 0)

        for idx, img_path in enumerate(img_paths, start=1):
            img = make_circle(Image.open(img_path).resize((150, 150)))
            img_tk = ImageTk.PhotoImage(img)
            image_refs.append(img_tk)
            canvas.create_image(x_offset, 10, anchor="nw", image=img_tk)
            x_offset += overlap

            pred_class1 = predict_image(model1, img_path, grayscale=False)
            pred_class2 = predict_image(model2, img_path, grayscale=False)
            results_text += f"Image {idx}: {img_path.split('/')[-1]}\n"
            results_text += f"VGG16 Prediction: {class_names[pred_class1]}, Mobilenet Prediction: {class_names[pred_class2]}\n\n"

        result_label.config(text=results_text)


def make_circle(img, size=(120, 120)):
    """
    Takes an image and turns it into a circle of given size.

    Parameters
    ----------
    img : PIL.Image
        Image to be turned into a circle.
    size : tuple
        Size of the circle to be created (default is (120, 120)).

    Returns
    -------
    PIL.Image
        Image with circular alpha mask.
    """

    img = img.resize(size)
    draw = Image.new("L", size, 0)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x - size[0] // 2) ** 2 + (y - size[1] // 2) ** 2 < (size[0] // 2) ** 2:
                draw.putpixel((x, y), 255)
    img.putalpha(draw)
    return img


# List for image references
image_refs = []

# Create the GUI
root = tk.Tk()
root.title("🧠 Brain MRI Classifier")
root.geometry("600x550")

# Font settings
TITLE_FONT = ("Helvetica", 24, "bold")
TEXT_FONT = ("Helvetica", 14)
BUTTON_FONT = ("Helvetica", 14, "bold")

# Title label
title_label = tk.Label(root, text="Brain MRI Classifier", font=TITLE_FONT)
title_label.pack(pady=(80, 0))

# Canvas
canvas = tk.Canvas(root, width=500, height=140, highlightthickness=0)
canvas.pack(pady=20)


result_label = Label(
    root,
    text="Select images to get predictions",
    font=TEXT_FONT,
    wraplength=500,
    justify="left",
)
result_label.pack(pady=10)

# Button to open file dialog
button = tk.Button(root, text="Select Images", font=BUTTON_FONT, command=open_images)
button.pack(pady=10)

root.mainloop()
