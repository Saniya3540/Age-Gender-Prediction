import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("age_gender_model.keras")

gender_dict = {0: 'Male', 1: 'Female'}

st.title("Age and Gender Prediction")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert('L')

    # Resize image
    image = image.resize((128, 128))

    # Convert to numpy array
    img = np.array(image)

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1, 128, 128, 1)

    # Predict
    pred = model.predict(img)

    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    # Show image
    st.image(uploaded_file, caption="Uploaded Image")

    # Results
    st.success(f"Predicted Gender: {pred_gender}")
    st.success(f"Predicted Age: {pred_age}")