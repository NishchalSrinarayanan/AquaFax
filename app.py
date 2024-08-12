import streamlit as st
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import openai
import io

# IBM Watson Visual Recognition credentials
api_key = st.secrets["ibm_api_key"]
service_url = st.secrets["ibm_service_url"]

# Set up IBM Watson Visual Recognition
authenticator = IAMAuthenticator(api_key)
visual_recognition = VisualRecognitionV3(
    version='2021-03-31',
    authenticator=authenticator
)
visual_recognition.set_service_url(service_url)

# Set your OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

st.title('Sea Animal Identifier')

# Upload an image
uploaded_file = st.file_uploader("Upload an image of a sea animal:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Convert the uploaded file to a binary stream
    image_binary = uploaded_file.read()

    # Use IBM Watson to classify the image
    classes = visual_recognition.classify(
        images_file=io.BytesIO(image_binary),
        threshold='0.6'
    ).get_result()

    # Get the top class
    if classes and 'images' in classes and 'classifiers' in classes['images'][0]:
        top_class = classes['images'][0]['classifiers'][0]['classes'][0]['class']
        st.write(f"Identified Sea Animal: {top_class}")

        # Use GPT-4 Turbo to get more details
        prompt = f"Tell me about the sea animal called '{top_class}'. Include its scientific name, whether it is endangered or not, conservation tips if it is endangered, and a few fun facts."
        response = openai.Completion.create(
            engine="gpt-4o-mini",  # Specify GPT-4 Turbo model
            prompt=prompt,
            max_tokens=400,
            temperature=0.5,
        )
        st.write(response.choices[0].text.strip())
    else:
        st.write("Sorry, I couldn't identify the sea animal in the image.")

st.write("This app uses IBM Watson's Visual Recognition to identify sea animals and OpenAI's GPT-4 Turbo to provide detailed information.")
