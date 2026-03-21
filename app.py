# app.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid
import os
import tensorflow as tf
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "fruits_360.keras")

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
        labels = {
            0: 'Almonds 1', 1: 'Apple 10', 2: 'Apple 11', 3: 'Apple 12', 4: 'Apple 13', 
            5: 'Apple 14', 6: 'Apple 17', 7: 'Apple 18', 8: 'Apple 19', 9: 'Apple 5', 
            10: 'Apple 6', 11: 'Apple 7', 12: 'Apple 8', 13: 'Apple 9', 14: 'Apple Braeburn 1', 
            15: 'Apple Core 1', 16: 'Apple Crimson Snow 1', 17: 'Apple Golden 1', 18: 'Apple Golden 2', 
            19: 'Apple Golden 3', 20: 'Apple Granny Smith 1', 21: 'Apple Pink Lady 1', 22: 'Apple Red 1', 
            23: 'Apple Red 2', 24: 'Apple Red 3', 25: 'Apple Red Delicious 1', 26: 'Apple Red Yellow 1', 
            27: 'Apple Red Yellow 2', 28: 'Apple Rotten 1', 29: 'Apple hit 1', 30: 'Apple worm 1', 
            31: 'Apricot 1', 32: 'Avocado 1', 33: 'Avocado Black 1', 34: 'Avocado Black 2', 
            35: 'Avocado Green 1', 36: 'Avocado ripe 1', 37: 'Banana 1', 38: 'Banana 3', 
            39: 'Banana 4', 40: 'Banana Lady Finger 1', 41: 'Banana Red 1', 42: 'Beans 1', 
            43: 'Beetroot 1', 44: 'BlackBerry 3', 45: 'Blackberrie 1', 46: 'Blackberrie 2', 
            47: 'Blackberrie half rippen 1', 48: 'Blackberrie not rippen 1', 49: 'Blueberry 1', 
            50: 'Cabbage red 1', 51: 'Cabbage white 1', 52: 'Cactus fruit 1', 53: 'Cactus fruit green 1', 
            54: 'Cactus fruit red 1', 55: 'Caju seed 1', 56: 'Cantaloupe 1', 57: 'Cantaloupe 2', 
            58: 'Cantaloupe 3', 59: 'Carambula 1', 60: 'Carrot 1', 61: 'Cauliflower 1', 
            62: 'Cherimoya 1', 63: 'Cherry 1', 64: 'Cherry 2', 65: 'Cherry 3', 66: 'Cherry 4', 
            67: 'Cherry 5', 68: 'Cherry Rainier 1', 69: 'Cherry Rainier 2', 70: 'Cherry Rainier 3', 
            71: 'Cherry Sour 1', 72: 'Cherry Wax Black 1', 73: 'Cherry Wax Red 1', 74: 'Cherry Wax Red 2', 
            75: 'Cherry Wax Red 3', 76: 'Cherry Wax Yellow 1', 77: 'Cherry Wax not ripen 1', 
            78: 'Cherry Wax not ripen 2', 79: 'Chestnut 1', 80: 'Clementine 1', 81: 'Cocos 1', 
            82: 'Corn 1', 83: 'Corn Husk 1', 84: 'Cucumber 1', 85: 'Cucumber 10', 86: 'Cucumber 11', 
            87: 'Cucumber 3', 88: 'Cucumber 4', 89: 'Cucumber 5', 90: 'Cucumber 6', 91: 'Cucumber 7', 
            92: 'Cucumber 8', 93: 'Cucumber 9', 94: 'Cucumber Ripe 1', 95: 'Cucumber Ripe 2', 
            96: 'Dates 1', 97: 'Eggplant 1', 98: 'Eggplant long 1', 99: 'Fig 1', 100: 'Ginger 2', 
            101: 'Ginger Root 1', 102: 'Gooseberry 1', 103: 'Granadilla 1', 104: 'Grape Blue 1', 
            105: 'Grape Pink 1', 106: 'Grape White 1', 107: 'Grape White 2', 108: 'Grape White 3', 
            109: 'Grape White 4', 110: 'Grape not ripen 1', 111: 'Grapefruit Pink 1', 112: 'Grapefruit White 1', 
            113: 'Guava 1', 114: 'Hazelnut 1', 115: 'Huckleberry 1', 116: 'Kaki 1', 117: 'Kiwi 1', 
            118: 'Kohlrabi 1', 119: 'Kumquats 1', 120: 'Lemon 1', 121: 'Lemon Meyer 1', 122: 'Limes 1', 
            123: 'Lychee 1', 124: 'Mandarine 1', 125: 'Mango 1', 126: 'Mango Red 1', 127: 'Mangostan 1', 
            128: 'Maracuja 1', 129: 'Melon Piel de Sapo 1', 130: 'Mulberry 1', 131: 'Nectarine 1', 
            132: 'Nectarine Flat 1', 133: 'Nectarine Flat 2', 134: 'Nut 1', 135: 'Nut 2', 136: 'Nut 3', 
            137: 'Nut 4', 138: 'Nut 5', 139: 'Nut Forest 1', 140: 'Nut Pecan 1', 141: 'Onion 2', 
            142: 'Onion Red 1', 143: 'Onion Red 2', 144: 'Onion Red Peeled 1', 145: 'Onion White 1', 
            146: 'Onion White Peeled 1', 147: 'Orange 1', 148: 'Papaya 1', 149: 'Papaya 2', 
            150: 'Passion Fruit 1', 151: 'Peach 1', 152: 'Peach 2', 153: 'Peach 3', 154: 'Peach 4', 
            155: 'Peach 5', 156: 'Peach 6', 157: 'Peach Flat 1', 158: 'Peanut shell 1x 1', 
            159: 'Pear 1', 160: 'Pear 10', 161: 'Pear 11', 162: 'Pear 12', 163: 'Pear 13', 
            164: 'Pear 2', 165: 'Pear 3', 166: 'Pear 5', 167: 'Pear 6', 168: 'Pear 7', 
            169: 'Pear 8', 170: 'Pear 9', 171: 'Pear Abate 1', 172: 'Pear Forelle 1', 173: 'Pear Kaiser 1', 
            174: 'Pear Monster 1', 175: 'Pear Red 1', 176: 'Pear Stone 1', 177: 'Pear Williams 1', 
            178: 'Pear common 1', 179: 'Pepino 1', 180: 'Pepper 2', 181: 'Pepper Green 1', 
            182: 'Pepper Orange 1', 183: 'Pepper Orange 2', 184: 'Pepper Red 1', 185: 'Pepper Red 2', 
            186: 'Pepper Red 3', 187: 'Pepper Red 4', 188: 'Pepper Red 5', 189: 'Pepper Yellow 1', 
            190: 'Physalis 1', 191: 'Physalis with Husk 1', 192: 'Pineapple 1', 193: 'Pineapple Mini 1', 
            194: 'Pistachio 1', 195: 'Pitahaya Red 1', 196: 'Plum 1', 197: 'Plum 2', 198: 'Plum 3', 
            199: 'Plum 4', 200: 'Plum hole 1', 201: 'Pomegranate 1', 202: 'Pomelo Sweetie 1', 
            203: 'Potato Red 1', 204: 'Potato Red Washed 1', 205: 'Potato Sweet 1', 206: 'Potato White 1', 
            207: 'Quince 1', 208: 'Quince 2', 209: 'Quince 3', 210: 'Quince 4', 211: 'Rambutan 1', 
            212: 'Raspberry 1', 213: 'Raspberry 2', 214: 'Raspberry 3', 215: 'Raspberry 4', 
            216: 'Raspberry 5', 217: 'Raspberry 6', 218: 'Redcurrant 1', 219: 'Salak 1', 
            220: 'Strawberry 1', 221: 'Strawberry 2', 222: 'Strawberry 3', 223: 'Strawberry Wedge 1', 
            224: 'Tamarillo 1', 225: 'Tangelo 1', 226: 'Tomato 1', 227: 'Tomato 10', 228: 'Tomato 2', 
            229: 'Tomato 3', 230: 'Tomato 4', 231: 'Tomato 5', 232: 'Tomato 7', 233: 'Tomato 8', 
            234: 'Tomato 9', 235: 'Tomato Cherry Maroon 1', 236: 'Tomato Cherry Orange 1', 
            237: 'Tomato Cherry Red 1', 238: 'Tomato Cherry Red 2', 239: 'Tomato Cherry Yellow 1', 
            240: 'Tomato Heart 1', 241: 'Tomato Maroon 1', 242: 'Tomato Maroon 2', 243: 'Tomato Yellow 1', 
            244: 'Tomato not Ripen 1', 245: 'Walnut 1', 246: 'Watermelon 1', 247: 'Zucchini 1', 
            248: 'Zucchini Green 1', 249: 'Zucchini dark 1'
        }
        predicted_label = labels[predicted_class]

        st.write(f"Predicted Fruit: {predicted_label}")

st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")