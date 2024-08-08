import openai
import streamlit as st
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image
import io

# Set up Google Cloud Vision API Client
vision_client = vision.ImageAnnotatorClient()

# Set your OpenAI API key (store securely in Streamlit secrets)
openai.api_key = st.secrets["openai_api_key"]

st.title("Species Information App (Google Vision + GPT-4o-mini)")
st.write("Upload an image to get information about the species.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the uploaded image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Use Google Cloud Vision API to classify the image
    image = types.Image(content=img_byte_arr)
    response = vision_client.label_detection(image=image)
    labels = response.label_annotations

    if labels:
        top_label = labels[0].description
        confidence = labels[0].score
        st.write(f"Predicted: {top_label} with a confidence of {confidence:.2f}")

        # Query GPT-4o-mini for species information
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
    else:
        st.write("No labels detected. Please try a different image.")
