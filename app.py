import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import openai
import streamlit as st
from PIL import Image

# Set your OpenAI API key here (it's better to handle this securely)
openai.api_key = st.secrets["openai_api_key"]

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

st.title("Species Information App")
st.write("Upload an image to get information about the species.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make predictions
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Extract the top result
    top_label = decoded_predictions[0][1]
    top_score = decoded_predictions[0][2]

    st.write(f"Predicted: {top_label} with a confidence of {top_score:.2f}")

    # Query OpenAI for species information
    def get_species_info(species_name):
        messages = [
            {"role": "system", "content": "You are an assistant providing detailed information about species."},
            {"role": "user", "content": (f"Provide concise information about the following species: {species_name}. "
                                          "Include its scientific name, ocean layer (if applicable), endangered or invasive status, "
                                          "and tips for conservation if it is endangered. Also, add a few fun facts.")}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use the correct model name
            messages=messages,
            max_tokens=400
        )

        return response.choices[0].message['content'].strip()

    # Get species information
    species_info = get_species_info(top_label)

    st.write("\nSpecies Information:")
    st.write(species_info)
