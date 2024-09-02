import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
import openai

# Set up OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Functions
def get_full_name(sea_animal_name):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please provide the Wikipedia page name for '{sea_animal_name}' formatted with underscores, so it can be used in the following URL: https://en.wikipedia.org/api/rest_v1/page/summary/. Do not add anything else to your response."}
        ],
        max_tokens=50
    )
    return response['choices'][0]['message']['content'].strip()

def get_wikipedia_summary(wikiname):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wikiname}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('extract', "Sorry, no summary available.")
    else:
        return "Sorry, I couldn't find information on this sea animal."

def get_chatgpt_details(sea_animal_name):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Provide 3 interesting facts about {sea_animal_name}, whether it is endangered or not, and specific conservation tips that middle-class people can do at home. make sure this response does not exceed 200 tokens."}
        ],
        max_tokens=300
    )
    return response['choices'][0]['message']['content'].strip()

# Load image classification model
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Streamlit UI
st.set_page_config(page_title="AquaFax Sea Animal Identifier", layout="wide")
st.title("🦑 AquaFax Sea Animal Identifier 🐠")

st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Upload an image of a sea animal", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify the image
    st.sidebar.text("Classifying image...")
    predictions = classifier(image)
    sea_animal_name = predictions[0]['label'].lower()

    st.sidebar.text("Fetching information...")

    # Display identified sea animal
    st.subheader(f"Identified as: **{sea_animal_name.title()}**")

    # Fetch the Wikipedia-compatible name and summary
    wikiname = get_full_name(sea_animal_name)
    summary = get_wikipedia_summary(wikiname)
    st.subheader("🐟 Animal Summary 🐟")
    st.write(summary)

    # Fetch additional details using ChatGPT
    chatgpt_summary = get_chatgpt_details(sea_animal_name)
    st.subheader("🌍 Additional Details and Conservation Tips 🌍")
    st.write(chatgpt_summary)

else:
    st.info("Upload an image of a sea animal to get started!")
