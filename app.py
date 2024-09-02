
from transformers import pipeline
from PIL import Image
import requests
import openai
import streamlit.components.v1 as components

# Set up OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Functions
def get_full_name(sea_animal_name):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Provide 3 interesting facts about {sea_animal_name}, whether it is endangered or not, and specific conservation tips that middle-class people can do at home. make sure this response does not exceed 200 tokens."}
        ],
        max_tokens=300
    )
    return response['choices'][0]['message']['content'].strip()

# Load image classification model
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Custom HTML/CSS/JavaScript for splash animation
splash_animation_html = """
<style>
.splash-container {
    position: relative;
    display: inline-block;
}

.splash {
    position: absolute;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0) 70%);
    transform: scale(0);
    opacity: 0.7;
    pointer-events: none;
    transition: transform 0.4s, opacity 0.4s;
}

.splash.active {
    transform: scale(1);
    opacity: 0;
}
</style>
<script>
function addSplashEffect(event) {
    var splash = document.createElement("div");
    splash.className = "splash";
    event.target.appendChild(splash);
    setTimeout(() => {
        splash.classList.add("active");
        setTimeout(() => {
            event.target.removeChild(splash);
        }, 400);
    }, 0);
}
document.querySelectorAll(".splash-button").forEach(button => {
    button.addEventListener("click", addSplashEffect);
});
</script>
"""

# Streamlit UI
st.set_page_config(page_title="AquaFax Sea Animal Identifier", layout="wide")

# Custom CSS for advanced styling
st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
    }
    .header {
        font-size: 1.5em;
        font-weight: bold;
        color: #2E8B57;
    }
    .subheader {
        font-size: 1.2em;
        font-weight: bold;
        color: #FF4500;
    }
    .info {
        font-size: 1em;
        color: #4682B4;
    }
    .image-container {
        display: flex;
        justify-content: center;
    }
    .summary-box {
        border: 2px solid #FFD700;
        padding: 15px;
        border-radius: 10px;
        background-color: #F0F8FF;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 0.9em;
        color: #696969;
    }
    .splash-button {
        position: relative;
        overflow: hidden;
        border: 2px solid #1E90FF;
        background-color: #87CEFA;
        color: #FFFFFF;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 1em;
        cursor: pointer;
        outline: none;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Display title
st.markdown('<div class="title">🦑 AquaFax Sea Animal Identifier 🐠</div>', unsafe_allow_html=True)

# Custom splash effect
components.html(splash_animation_html, height=0, width=0)

# Sidebar with splash effect
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Upload an image of a sea animal", type=["jpg", "jpeg", "png"])

# Button to start processing
if st.sidebar.button("Process Image", key="process_image", help="Process the uploaded image to identify the sea animal"):
    if uploaded_file:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Classify the image
        st.sidebar.text("Classifying image...")
        predictions = classifier(image)
        sea_animal_name = predictions[0]['label'].lower()

        # Show results
        st.markdown(f'<div class="subheader">Identified as: **{sea_animal_name.title()}**</div>', unsafe_allow_html=True)
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="header">🐟 Animal Summary 🐟</div>', unsafe_allow_html=True)
        wikiname = get_full_name(sea_animal_name)
        summary = get_wikipedia_summary(wikiname)
        st.write(summary)
        st.markdown('</div>', unsafe_allow_html=True)

        # Fetch additional details using ChatGPT
        chatgpt_summary = get_chatgpt_details(sea_animal_name)
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="header">🌍 Additional Details and Conservation Tips 🌍</div>', unsafe_allow_html=True)
        st.write(chatgpt_summary)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Upload an image of a sea animal to get started!")
