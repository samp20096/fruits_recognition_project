import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the model and image based on the script directory
MODEL_PATH = os.path.join(SCRIPT_DIR, "fruits_360.keras")
image_path = os.path.join(SCRIPT_DIR, "saved_images", "banana01.jpg")

model = tf.keras.models.load_model(MODEL_PATH)

def upload_img(file_obj):
    img = Image.open(file_obj)
    img = img.resize((100, 100))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

prediction = model.predict(upload_img(image_path))
predicted_class = np.argmax(prediction[0])
print(f"Predicted class index: {predicted_class}")