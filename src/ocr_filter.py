from paddleocr import PaddleOCR
import cv2
import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Ocr_filter:
    """
    Detect screw standard text within image
    """
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, enable_mkldnn=True)
        self.standard_list = ['M3', 'M4', 'M5', 'M6', 'M8', 'M10']
        # Compile a regular expression that matches any of the standards in self.standard_list (case-insensitive)
        self.pattern = re.compile('|'.join([re.escape(standard) for standard in self.standard_list]), re.IGNORECASE)
    
    def split_image(self, image, tile_size=(1200, 1200), overlap=0.2):
        """
        Splits the image into smaller tiles, with optional overlap.

        Args:
            image (np.ndarray): The image to split.
            tile_size (tuple): The desired tile size (height, width).
            overlap (float): The overlap ratio between adjacent tiles (0 to 1).

        Returns:
            List of image tiles.
        """
        img_height, img_width = image.shape[:2]
        step_height = int(tile_size[0] * (1 - overlap))
        step_width = int(tile_size[1] * (1 - overlap))
        
        tiles = []
        for y in range(0, img_height, step_height):
            for x in range(0, img_width, step_width):
                x_end = min(x + tile_size[1], img_width)
                y_end = min(y + tile_size[0], img_height)
                tile = image[y:y_end, x:x_end]
                tiles.append((tile, x, y))  # Store tile and its position (x, y) for later adjustment
        return tiles

    def find_standard_and_scale(self, image_path):
        """
        Splits the image into tiles and runs OCR on each tile, then combines the results.

        Args:
            image_path (str): Path to the image.

        Returns:
            List of detected standards.
        """
        image = cv2.imread(image_path)
        detected_standard = set()  # Use set to avoid duplicates and faster duplicate check compared to list loop, list loop to find duplicate is O(n), set is O(1) on average.
        scale = set()  # Same reason as above
        scale_pattern = r"\b\d+\s?:\s?\d+\b" #Pattern of the scale value ([num]:[num] e.g., 1:1, 1:2)

        tiles = self.split_image(image)

        for tile, x_offset, y_offset in tiles:
            result = self.ocr.ocr(tile)
            for line in result:
                if line:
                    for word_info in line:
                        if word_info and word_info[1]:
                            text = word_info[1][0]
                            # Check if any standard pattern is matched in the text
                            matches = self.pattern.findall(text)  # Find all matched standards in the text
                            for match in matches:
                                detected_standard.add(match.upper())  # Add the match to the set (no duplicates)

                            # Check for scale pattern (e.g., "1:1", "2:3", etc.)
                            scale_match = re.match(scale_pattern, text)
                            if scale_match:
                                scale_value = scale_match.group(0)  # Extract the matched scale value
                                scale_value = scale_value.replace(" ", "")  # Remove all spaces
                                scale.add(scale_value)  # Add the scale value to the set (no duplicates)

        return list(detected_standard), list(scale)