import os
import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

EXAMPLE_IMAGES_PATH = "./images/"
SAVED_MODEL_PATH = "./model/nn.h5"
N_CATEGORIES = 5
CATEGORIES = ("Coast", "Desert", "Forest", "Glacier", "Mountain")
IMAGE_SIZE = (200, 150)
MODEL = tf.keras.models.load_model(SAVED_MODEL_PATH)


st.set_page_config(page_title="Landscape classification", page_icon="🌍", layout="centered")
st.title("Landscape classification")

st.subheader("Categories")
cols = st.columns(len(os.listdir(EXAMPLE_IMAGES_PATH)))

for i, image_path in enumerate(os.listdir(EXAMPLE_IMAGES_PATH)):
    image = cv2.imread(os.path.join(EXAMPLE_IMAGES_PATH, image_path))
    image = cv2.resize(image, IMAGE_SIZE)
    cols[i].image(image, caption=image_path.split(".")[0])

image = st.file_uploader(
    "Upload your image here", 
    accept_multiple_files=False,
    type=["jpg", "jpeg", "png"]
)

if image:
    st.image(image, caption="Image uploaded", use_column_width=True)

    if st.button("Predict"):
        cv2_image = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        cv2_image = cv2.resize(cv2_image, (150, 150))
        cv2_image = np.reshape(cv2_image, (1, 150, 150, 3))
        prediction = MODEL(cv2_image)
        st.success(f"Value Predicted: {CATEGORIES[np.argmax(prediction)]}")