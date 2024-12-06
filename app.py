from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
import cv2
import numpy as np
import src.image_processing
import tempfile

app = FastAPI()

@app.post("/process-image/")
async def process_image(image: UploadFile = File(...)):
    # Save the uploaded image to a temporary file
    post_file = image.file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(post_file)

    print(tmp_file.name)
    weight_path = "weights/best.pt"
    # Process image and weights
    filtered_classifications_json, scale_json, filtered_classifications_all, x1y1_list_all, x2y2_list_all = src.image_processing.process_image(tmp_file.name, weight_path)
    
    # Get the list of modified images and the corresponding scale values
    image_list, scale_list = src.image_processing.save_bounding_boxes_for_all_scales(tmp_file.name, x1y1_list_all, x2y2_list_all, filtered_classifications_all)
    
    # Convert each processed image into a byte stream
    image_streams = []
    for processed_img in image_list:
        _, img_encoded = cv2.imencode('.jpg', processed_img)
        img_bytes = img_encoded.tobytes()
        img_stream = BytesIO(img_bytes)
        image_streams.append(img_stream)
    
    # Prepare the response with JSON data and the list of image streams
    response = {
        "filtered_classifications_json": filtered_classifications_json,
        "scale_json": scale_json,
        "images": []
    }
    
    # Adding image streams to the response (Note: images are returned as base64-encoded strings in JSON)
    for idx, img_stream in enumerate(image_streams):
        # Convert the image stream to base64 and append it to the response
        import base64
        img_base64 = base64.b64encode(img_stream.getvalue()).decode("utf-8")
        response["images"].append({
            "image_id": idx,
            "image_base64": img_base64
        })
    
    return JSONResponse(content=response)

# Run the application (this can be done from command line via `uvicorn script_name:app --reload`)