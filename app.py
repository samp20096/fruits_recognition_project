# app.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid
import os
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "fruits_360.keras"

if not os.path.exists(MODEL_PATH):
    """
    This is a simple check to make sure the model file is in the same folder as app.py
    If the model file is not found, the app will not run
    """
    st.warning("The system is currently loading the model")
    st.error("Model file not found. Please check that file is in the same folder as app.py")
    st.stop()

@st.cache_resource
def load_model():
    """
    This function loads the model from the model.keras file
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


st.set_page_config(page_title="Upload Photo", page_icon="📷")

st.title("📷 Upload a Photo (Camera or Gallery)")
st.write("Take a photo or upload one, then save it on the computer running this app")

# Folder where images will be saved (on the computer/server)
SAVE_DIR = Path("saved_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_image(file_obj, prefix: str = "img") -> Path:
    """
    Saves a Streamlit UploadedFile-like object to disk and returns the saved path
    """
    # Try to keep the original extension if present
    original_name = getattr(file_obj, "name", "") or ""
    ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique}{ext}"
    out_path = SAVE_DIR / filename

    # Read bytes and write to file
    data = file_obj.getvalue()
    out_path.write_bytes(data)
    return out_path

st.subheader("1) Take a picture (phone camera)")
camera_photo = st.camera_input("Open camera and take a photo")

st.subheader("2) Or upload from gallery")
gallery_photo = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

chosen = camera_photo if camera_photo is not None else gallery_photo

if chosen is not None:
    st.image(chosen, caption="Preview", use_container_width=True)

    if st.button("💾 Save image to computer"):
        try:
            saved_path = save_uploaded_image(chosen, prefix="phone")
            st.success(f"Saved ✅  {saved_path.resolve()}")
            st.info("The image was saved on the computer running Streamlit (server)")
        except Exception as e:
            st.error(f"Failed to save: {e}")


    if st.button("Fruit Recognition"):
        img = Image.open(chosen)
        img = img.resize((100, 100))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction[0])
        predicted_label = model.classes_[predicted_class]

        st.write(f"Predicted Fruit: {predicted_label}")

st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")