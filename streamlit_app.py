import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import json

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
        
        # Send the uploaded image to FastAPI
        files = {"image": uploaded_file.getvalue()}
        try:
            response = requests.post(FASTAPI_URL, files=files)
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses

            # Process response
            response_json = response.json()
            filtered_classifications_json = response_json.get("filtered_classifications_json")
            scale_json = response_json.get("scale_json")
            images_base64 = response_json.get("images", [])

            # Handle 'filtered_classifications_json' as a string (if it's a JSON string)
            if isinstance(filtered_classifications_json, str):
                try:
                    filtered_classifications_json = json.loads(filtered_classifications_json)
                except json.JSONDecodeError:
                    st.error("Error decoding the filtered classifications JSON string.")
                    return

            # Display the filtered classifications without JSON formatting
            st.subheader("Filtered Classifications")
            if filtered_classifications_json:
                for scale, classifications in filtered_classifications_json.items():
                    st.write(f"Scale: {scale}")
                    for category, count in classifications.items():
                        st.write(f"{category}: {count}")

            # Handle 'scale_json' as a string (if it's a JSON string)
            if isinstance(scale_json, str):
                try:
                    scale_json = json.loads(scale_json)
                except json.JSONDecodeError:
                    st.error("Error decoding the scale JSON string.")
                    return

            # Display the scale without JSON formatting
            st.subheader("Scale")
            if scale_json:
                if isinstance(scale_json, dict) and "scale" in scale_json:
                    st.write("Scale values:")
                    for scale in scale_json["scale"]:
                        st.write(f"- {scale}")
                elif isinstance(scale_json, list):
                    st.write("Scale values:")
                    for scale in scale_json:
                        st.write(f"- {scale}")
                else:
                    st.error("Unexpected format for scale_json.")

            # Display the processed images
            st.subheader("Processed Images")
            if images_base64:
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
