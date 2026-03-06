import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "fruits_360.keras"

model = tf.keras.models.load_model(MODEL_PATH)

image_path = "saved_images/phone_20260222_001513_8185bd7f.webp"

def upload_img(file_obj):
    img = Image.open(file_obj)
    img = img.resize((100, 100))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

model.predict(upload_img(file_obj))