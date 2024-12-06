import cv2
from YOLO_with_slicer import YOLO_with_slicer 
from ocr_filter import Ocr_filter
import hole_classification
from collections import Counter
import json

KNN_THRESHOLD = 1.0

def process_image(image_path, weight_path, KNN_THRESHOLD=1.0):
    """
    Processes an image to detect holes, classify them using YOLO and KNN, and outputs 
    the classification counts and scale information in JSON format.

    This function reads an image, detects holes using YOLO, converts the bounding boxes from pixels to 
    millimeters, classifies the holes using KNN, filters classifications based on OCR detected standards, 
    and outputs the results as JSON.

    Args:
        image_path (str): Path to the image to process.
        weight_path (str, optional): Path to the YOLO model weights file.
        KNN_THRESHOLD (float, optional): Threshold for KNN classification (default is 1.0).

    Returns:
        tuple: A tuple containing the following elements:
            - `filtered_classifications_json`: JSON-formatted classification counts for each scale.
            - `scale_json`: JSON-formatted scale data (a list of scales).
            - `filtered_classifications_by_scale`: Dictionary of filtered classifications for each scale.
            - `x1y1_list_all`: List of lists of top-left corner coordinates for bounding boxes for each scale.
            - `x2y2_list_all`: List of lists of bottom-right corner coordinates for bounding boxes for each scale.

    Output Format:
        1. `filtered_classifications_json`: A JSON object where each scale is a key, and its value is a dictionary with classification counts:
        
            Example:
            ```json
            {
                "1:3": {
                    "M4": 22,
                    "M5": 23,
                    "M6": 13,
                    "Unknown": 26
                },
                "1:4": {
                    "M4": 8,
                    "M5": 8,
                    "M6": 20,
                    "Unknown": 48
                }
            }
            ```

        2. `scale_json`: A JSON object containing a list of scales:

            Example:
            ```json
            {
                "scale": [
                    "1:3",
                    "1:4"
                ]
            }
            ```
        
        3. `filtered_classifications_by_scale`: A dictionary where each scale key maps to a list of classifications for that scale (pre-filtered based on OCR detected standards).

        4. `x1y1_list_all`: A list of lists, where each inner list contains the top-left corner coordinates `(x1, y1)` for bounding boxes at each scale.
        
        5. `x2y2_list_all`: A list of lists, where each inner list contains the bottom-right corner coordinates `(x2, y2)` for bounding boxes at each scale.
    """
    # Initialize YOLO with slicer
    yolo_slicer = YOLO_with_slicer(weight_path)

    # Get predictions using YOLO
    predictions = yolo_slicer.predict(image_path)

    # Read and process image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary

    # Get image dimensions and DPI
    width, height, dpi = hole_classification.get_width_height_dpi(image_path)

    # Initialize OCR filter and get detected standard and scale
    ocr = Ocr_filter()
    detected_standard, scale_list = ocr.find_standard_and_scale(image_path)

    # Initialize lists for bounding box data
    mm_widths_img_all = []
    mm_heights_img_all = []
    x1y1_list_all = []
    x2y2_list_all = []

    # Process each detection
    for scale in scale_list:
        x1y1_list = []
        x2y2_list = []
        mm_widths_img = []
        mm_heights_img = []
        for result in predictions:
            cls, score, bbox = result[0], result[1], result[2]
            if score >= 0.40:
                # Convert bounding box to mm and pixel coordinates
                width_mm, height_mm, x1y1, x2y2 = hole_classification.get_bounding_box_mm_and_pixel(
                    bbox, scale, dpi, width, height)

                mm_widths_img.append(width_mm)
                mm_heights_img.append(height_mm)
                x1y1_list.append(x1y1)
                x2y2_list.append(x2y2)
        x1y1_list_all.append(x1y1_list)
        x2y2_list_all.append(x2y2_list)
        mm_widths_img_all.append(mm_widths_img)
        mm_heights_img_all.append(mm_heights_img)
        

    # Classify holes using KNN and create a dictionary for filtered classifications by scale
    filtered_classifications_by_scale = {}
    
    for i, scale in enumerate(scale_list):
        classifications = hole_classification.classify_holes_knn(mm_widths_img_all[i], mm_heights_img_all[i], distance_threshold=KNN_THRESHOLD)

        # Filter classifications based on detected standards
        filtered_classifications = [value if value in detected_standard else 'Unknown' for value in classifications]

        # Store the filtered classifications in the dictionary with the scale as the key
        filtered_classifications_by_scale[scale] = filtered_classifications

    # Count the occurrences of each classification for each scale
    classification_counts_by_scale = {}
    for scale, classifications in filtered_classifications_by_scale.items():
        # Use Counter to count the frequency of each classification
        classification_counts_by_scale[scale] = dict(Counter(classifications))

    # Prepare the JSON output for filtered classifications with counts
    filtered_classifications_json = json.dumps(classification_counts_by_scale, indent=4, sort_keys=True)

    # Prepare scale data JSON
    scale_data = {
        "scale": scale_list
    }
    scale_json = json.dumps(scale_data, indent=4, sort_keys=True)

    return filtered_classifications_json, scale_json, filtered_classifications_by_scale, x1y1_list_all, x2y2_list_all

def save_bounding_boxes_image(image_path, x1y1_list, x2y2_list, filtered_classifications):
    """
    This function draws bounding boxes on an image based on the provided coordinates
    and highlights them with colors corresponding to the classifications. The modified
    image is then returned as a result (instead of being saved).

    Args:
        image_path (str): Path to the image file on which bounding boxes will be plotted.
        x1y1_list (list of tuples): List of top-left corner coordinates (x1, y1) of bounding boxes.
        x2y2_list (list of tuples): List of bottom-right corner coordinates (x2, y2) of bounding boxes.
        filtered_classifications (list of str): List of class names for each bounding box (e.g., 'M3', 'M4', etc.).

    Returns:
        image (ndarray): The image with bounding boxes and classifications drawn on it.
    """
    
    # Color map to map classification names to specific colors (BGR format)
    def get_class_color(class_name):
        color_map = {
            'M3': (0, 128, 0),        # Dark Green
            'M4': (0, 0, 128),        # Dark Blue
            'M5': (128, 0, 0),        # Dark Red
            'M6': (255, 215, 0),      # Golden Yellow
            'M8': (0, 128, 128),      # Dark Cyan
            'M10': (128, 0, 128),     # Dark Magenta
            'Unknown': (105, 105, 105) # Dark Grey
        }
        return color_map.get(class_name, (0, 0, 0))  # Default to black if unknown class
    
    image = cv2.imread(image_path)
    
    # Loop through all the bounding boxes and classifications
    for i in range(len(x1y1_list)):
        x1, y1 = x1y1_list[i]
        x2, y2 = x2y2_list[i]
        classification = filtered_classifications[i]

        # Get the color for the current class
        color = get_class_color(classification)
        
        # Draw the bounding box (rectangle) around the object
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add the label (classification) to the image
        label = f"{classification}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 3
        cv2.putText(image, label, (x1, y1 - 10), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def save_bounding_boxes_for_all_scales(image_path, x1y1_list_all, x2y2_list_all, filtered_classifications_all):
    """
    This function iterates over all scales, modifies images with bounding boxes for each scale,
    and returns a list of modified images along with a list of corresponding scale values.

    Args:
        image_path (str): Path to the image file.
        x1y1_list_all (list of lists): List of lists of top-left corner coordinates (x1, y1) for each scale.
        x2y2_list_all (list of lists): List of lists of bottom-right corner coordinates (x2, y2) for each scale.
        filtered_classifications_all (dict): Dictionary where each key is a scale value, and the corresponding value
        is a list of classification names for each bounding box at that scale.

    Returns:
        tuple: A tuple containing:
            - `images_list` (list of ndarray): A list of modified images with bounding boxes drawn on them.
            - `scale_values` (list of str): A list of scale values corresponding to each image in `images_list`.
    """
    # Create a list to store modified images and scale values
    images_list = []
    scale_values = list(filtered_classifications_all.keys())  # Get the list of scale values directly from the keys

    # Iterate over each scale and process the image
    for i, scale in enumerate(filtered_classifications_all.keys()):
        # Get corresponding bounding boxes and classifications for the current scale
        x1y1_list = x1y1_list_all[i]
        x2y2_list = x2y2_list_all[i]
        filtered_classifications = filtered_classifications_all[scale]

        # Call the function to modify the image for the current scale
        modified_image = save_bounding_boxes_image(image_path, x1y1_list, x2y2_list, filtered_classifications)
        
        # Append the modified image to the list
        images_list.append(modified_image)

    # Return both the list of images and the list of scale values
    return images_list, scale_values
