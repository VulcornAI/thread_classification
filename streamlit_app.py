import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# FastAPI server URL
FASTAPI_URL = "http://localhost:8000/process-image/"

def main():
    st.title("Image Processing with FastAPI")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff", "TIF", "TIFF"])
    
    if uploaded_file:
        # Show image preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Send the uploaded image and weight file path to FastAPI
        files = {"image": uploaded_file.getvalue()}
        try:
            response = requests.post(FASTAPI_URL, files=files)
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses

            # Process response
            response_json = response.json()
            filtered_classifications_json = response_json.get("filtered_classifications_json")
            scale_json = response_json.get("scale_json")
            images_base64 = response_json.get("images", [])

            # Display JSON data
            st.subheader("Filtered Classifications JSON")
            st.json(filtered_classifications_json)

            st.subheader("Scale JSON")
            st.json(scale_json)

            # Display the processed images
            st.subheader("Processed Images")
            for img_data in images_base64:
                img_base64 = img_data.get("image_base64")
                img_id = img_data.get("image_id")
                
                # Decode the image from base64
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(BytesIO(img_bytes))
                st.image(img, caption=f"Processed Image {img_id}", use_column_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
