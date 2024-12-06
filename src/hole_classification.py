import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# Define the screw sizes and corresponding hole diameters in mm (based on the table)
screw_sizes = {
    "M3": 2.5,
    "M4": 3.3,
    "M5": 4.2,
    "M6": 5.0,
    "M8": 6.7,
    "M10": 8.5
}

Image.MAX_IMAGE_PIXELS = None
KNN_THRESHOLD = 1.0 

def get_width_height_dpi(image: str) -> tuple[int, int, int]:
    """
    Retrieves the dimensions (width and height) of an image in pixels and its DPI (dots per inch).

    Args:
        image (str or Path): The path to the image file.

    Returns:
        tuple: A tuple containing the image's width (int), height (int), and DPI (int).
            - width (int): The width of the image in pixels.
            - height (int): The height of the image in pixels.
            - dpi (int): The DPI (dots per inch) of the image. Defaults to 400 if not provided in image metadata.
    """
    with Image.open(image) as img:
        width, height = img.size
        dpi = img.info.get('dpi', (400, 400))[0]
        return width, height, dpi

def convert_pixels_to_mm(scale: tuple[float, float], dpi: float, pixel_width: int, pixel_height: int) -> tuple[float, float]:
    """
    Converts the pixel dimensions of an image to millimeters, applying a scale factor.

    Args:
        scale (tuple): The scale factor (x, y) that adjusts the final size.
            - x (float): Scale factor for width.
            - y (float): Scale factor for height.
        dpi (float): The DPI (dots per inch) of the image.
        pixel_width (int): The width of the image in pixels.
        pixel_height (int): The height of the image in pixels.

    Returns:
        tuple: A tuple containing the width and height in millimeters (float, float).
            - width_mm (float): The width of the image in millimeters, after scaling.
            - height_mm (float): The height of the image in millimeters, after scaling.
    """
    width_inch = pixel_width / dpi
    height_inch = pixel_height / dpi
    
    width_mm = width_inch * 25.4
    height_mm = height_inch * 25.4
    
    x, y = scale
    scale_multiplier = y/x
    width_mm *= scale_multiplier
    height_mm *= scale_multiplier
    
    return width_mm, height_mm

def classify_holes_knn(mm_widths: list[float], mm_heights: list[float], distance_threshold: float = 0.1) -> list[str]:
    """
    Classify detected holes using KNN based on screw size standards, with outlier rejection.

    Args:
        mm_widths (list of float): The list of hole widths in millimeters.
        mm_heights (list of float): The list of hole heights in millimeters.
        distance_threshold (float): The maximum distance for a hole to be classified. Holes with distances above this threshold are classified as "Unknown".

    Returns:
        list of str: A list of classifications for each detected hole. The classification is either a screw size label or "Unknown" if the hole does not match any known screw size.
    """
    # Prepare the feature matrix (width and height) and the labels (screw sizes)
    screw_sizes_list = list(screw_sizes.values())
    screw_labels = list(screw_sizes.keys())
    
    # Features: The screw diameters (since we assume the hole size is approximately the screw diameter)
    X_train = np.array([[diameter, diameter] for diameter in screw_sizes_list])  # (diameter, diameter) for simplicity
    y_train = screw_labels  # Corresponding screw sizes

    # Initialize the KNN classifier (with a reasonable k value)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Classify each detected hole
    predictions = []
    for width_mm, height_mm in zip(mm_widths, mm_heights):
        # Calculate the distance to each screw size in the training set
        distances, indices = knn.kneighbors([[width_mm, height_mm]])

        # Find the closest neighbor
        closest_distance = distances[0][0]
        if closest_distance > distance_threshold:
            # If the closest distance is larger than the threshold, mark as "Unknown"
            predictions.append("Unknown")
        else:
            # Otherwise, classify it based on the closest neighbor
            predicted_label = y_train[indices[0][0]]
            predictions.append(predicted_label)
    
    return predictions

def get_bounding_box_mm_and_pixel(bbox: tuple[float, float, float, float], scale: str, dpi: float, width: int, height: int) -> tuple[float, float, tuple[int, int], tuple[int, int]]:
    """
    Calculates the width and height of a bounding box in millimeters, using the provided scale and DPI.
    The scale is provided as a string (e.g., '1:1', '1:2') and needs to be parsed. Also, calculates the pixel coordinates 
    of the bounding box corners.

    Args:
        bbox (tuple of float): A tuple representing the bounding box coordinates (x1, y1, x2, y2), where each value is a float in the range [0, 1], representing normalized coordinates.
        scale (str): The scale factor in the form of a string like '1:1' or '1:2'. This defines the ratio of real-world size to image size.
        dpi (float): The DPI (dots per inch) to convert pixel values to millimeters.
        width (int): The width of the image or canvas in pixels.
        height (int): The height of the image or canvas in pixels.

    Returns:
        tuple of float: A tuple containing:
            - The width of the bounding box in millimeters.
            - The height of the bounding box in millimeters.
            - The pixel coordinates of the top-left corner of the bounding box (x1_pixel, y1_pixel).
            - The pixel coordinates of the bottom-right corner of the bounding box (x2_pixel, y2_pixel).
    """
    # Parse the scale ratio string (e.g., '1:1' or '1:2')
    scale_x, scale_y = map(int, scale.split(':'))  # Split by ':' and convert to integers

    # Extract coordinates from bbox
    x1, y1, x2, y2 = bbox
    
    # Convert the normalized coordinates to pixels based on the provided width and height
    x1_pixel = int(x1 * width)
    y1_pixel = int(y1 * height)
    x2_pixel = int(x2 * width)
    y2_pixel = int(y2 * height)

    # Calculate width and height in pixels
    width_pixels = x2_pixel - x1_pixel
    height_pixels = y2_pixel - y1_pixel
    
    # Convert the pixel dimensions to millimeters using the convert_pixels_to_mm function
    width_mm, height_mm = convert_pixels_to_mm((scale_x, scale_y), dpi, width_pixels, height_pixels)
    
    # Apply an offset to adjust the final dimensions
    width_mm -= 1.5
    height_mm -= 1.5
    
    return width_mm, height_mm, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel)
