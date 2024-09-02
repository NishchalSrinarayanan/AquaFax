import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
import openai

# Initialize the OpenAI API
openai.api_key = st.secrets["openai_api_key"]
# Function to get facts from Wikipedia
def get_wikipedia_summary(sea_animal_name):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{sea_animal_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('extract')
    else:
        return "Sorry, I couldn't find information on this sea animal."

# Function to get additional details using ChatGPT API
def get_chatgpt_details(sea_animal_name):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Specify the correct model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Provide some interesting facts about {sea_animal_name}. wether it is endangered or not, and if the animal is, some conservation tips for regular middle-class people to do at home. Be specific with the conservation tips."}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Load a pre-trained image classification model
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Streamlit UI
st.title("Sea Animal Identifier and Facts Fetcher")

# Upload picture
uploaded_file = st.file_uploader("Upload an image of a sea animal", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Use the image classifier to identify the sea animal or object
    predictions = classifier(image)

    # Get the top prediction and convert it to a lower case name for matching
    sea_animal = predictions[0]['label'].lower()

    st.write(f"Identified as: **{sea_animal}**")

    # Fetch facts from Wikipedia
    summary = get_wikipedia_summary(sea_animal)
    st.write(f"**Wikipedia Summary:** {summary}")
    
    # Fetch additional details using ChatGPT
    chatgpt_summary = get_chatgpt_details(sea_animal)
    st.write(f"**Additional Details from ChatGPT:** {chatgpt_summary}")
