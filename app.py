import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import os
import boto3

# take environment variables from .env
load_dotenv()

# Lambda URL
url = os.getenv("AWS_LAMBDA_URL")

# set title
st.title('Birds Classifier')

# set header
st.header('Upload an image of the bird')

# upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# display image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=False, width=300)

    # Save the image to a file
    image.save("temp.jpg", format="JPEG")

    # Upload the file to S3
    s3 = boto3.client('s3')
    with open("temp.jpg", "rb") as data:
        s3.upload_fileobj(data, 'birds-classification', 'myimage.jpg')

    # Now 'myimage.jpg' is accessible at 'https://mybucket.s3.amazonaws.com/myimage.jpg'
    img_url = 'https://birds-classification.s3.amazonaws.com/myimage.jpg'

    # Send the image URL as JSON data
    result = requests.post(url, json={"url": img_url})

    # write classification
    st.write(f"**Bird Category Prediction**: 👉👉👉{result.json()}👈👈👈")
