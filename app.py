import streamlit as st
import requests
import openai

# API credentials
deepai_api_key = st.secrets["deepai_api_key"]
openai_api_key = st.secrets["openai_api_key"]

# Set your OpenAI API key
openai.api_key = openai_api_key

st.title('Sea Animal Identifier')

# Upload an image
uploaded_file = st.file_uploader("Upload an image of a sea animal:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Send image to DeepAI for recognition
    response = requests.post(
        "https://api.deepai.org/api/image-recognition",
        files={"image": uploaded_file.getvalue()},
        headers={"api-key": deepai_api_key}
    )

    if response.status_code == 200:
        result = response.json()
        if 'output' in result and 'objects' in result['output']:
            objects = result['output']['objects']
            if objects:
                top_class = objects[0]['name']
                st.write(f"Identified Object: {top_class}")

                # Use GPT-3.5-turbo to get more details
                prompt = (f"Tell me about the sea animal called '{top_class}'. Include its scientific name, "
                          "whether it is endangered or not, conservation tips if it is endangered, and a few fun facts.")
                
                gpt_response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # Cheaper model than GPT-4
                    messages=[
                        {"role": "system", "content": "You are a marine biologist assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.5,
                )
                animal_info = gpt_response['choices'][0]['message']['content']
                st.write(animal_info)
            else:
                st.write("No identifiable objects found.")
        else:
            st.write("Sorry, I couldn't identify the objects in the image.")
    else:
        st.write("There was an error processing the image.")

st.write("This app uses DeepAI's image recognition to identify objects in images and OpenAI's GPT-3.5-turbo to provide detailed information.")
