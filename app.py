import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
import openai


openai.api_key = st.secrets["openai_api_key"]


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
            {"role": "user", "content": f"Provide 3 interesting facts about {sea_animal_name}, whether it is endangered or not, and specific conservation tips that middle-class people can do at home. make sure this response does not exceed 300 tokens."}
        ],
        max_tokens=300
    )
    return response['choices'][0]['message']['content'].strip()


classifier = pipeline("image-classification", model="google/vit-base-patch16-224")


st.title("AquaFax Sea Animal Identifier")


uploaded_file = st.file_uploader("Upload an image of a sea animal", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Use the image classifier to identify the sea animal or object
    predictions = classifier(image)

    # Get the top prediction and convert it to a lower case name for matching
    sea_animal_name = predictions[0]['label'].lower()

    st.write(f"Identified as: **{sea_animal_name}**")

    # Get the Wikipedia-compatible name for the sea animal
    wikiname = get_full_name(sea_animal_name)

    # Fetch facts from Wikipedia
    summary = get_wikipedia_summary(wikiname)
    st.write(f"**Animal Summary:** {summary}")
    
    # Fetch additional details using ChatGPT
    chatgpt_summary = get_chatgpt_details(sea_animal_name)
    st.write(f"**Additional Details And Conservation Tips:** {chatgpt_summary}")
