import argparse
import os
import src.image_processing
import cv2

def main(args):
    weight_path = "weights/best.pt"
    
    # Set default output_path based on image_path (using the base name of the image)
    output_path = os.path.splitext(args.image_path)[0] + '.jpg'

    # Process the image and get the results
    filtered_classifications_json, scale_json, filtered_classifications_all, x1y1_list_all, x2y2_list_all = src.image_processing.process_image(args.image_path, weight_path)

    # Get the list of modified images and the list of scale values
    image_list, scale_list = src.image_processing.save_bounding_boxes_for_all_scales(args.image_path, x1y1_list_all, x2y2_list_all, filtered_classifications_all)

    # Loop through the images and their corresponding scale values
    for idx, (img, scale) in enumerate(zip(image_list, scale_list)):
        # Construct a filename by appending the scale value to the output path
        scale_str = scale.replace(":", "-")  # Replace ':' with '-' in scale to avoid invalid characters in filenames
        output_filename = f"{output_path.split('.')[0]}_{scale_str}.jpg"
        
        # Save the image to the specified filename
        cv2.imwrite(output_filename, img)
        print(f"Image saved to {output_filename}")

    print(filtered_classifications_json)
    print(scale_json)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process image and save bounding boxes of holes detected.")

    # Define the command-line argument for image_path
    parser.add_argument('image_path', type=str, help='Path to the input image file')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)



# import src.image_processing
# import cv2

# # Set file paths
# output_path = "1.jpg"
# weight_path = 'weights/best.pt'
# image_path = 'D0067.tif'

# # Process the image and get the necessary data
# filtered_classifications_json, scale_json, filtered_classifications_all, x1y1_list_all, x2y2_list_all = src.image_processing.process_image(image_path, weight_path)

# # Get the list of modified images and the list of scale values
# image_list, scale_list = src.image_processing.save_bounding_boxes_for_all_scales(image_path, x1y1_list_all, x2y2_list_all, filtered_classifications_all)

# # Loop through the images and their corresponding scale values
# for idx, (img, scale) in enumerate(zip(image_list, scale_list)):
#     # Construct a filename by appending the scale value to the output path
#     scale_str = scale.replace(":", "-")  # Replace ':' with '-' in scale to avoid invalid characters in filenames
#     output_filename = f"{output_path.split('.')[0]}_{scale_str}.jpg"
    
#     # Save the image to the specified filename
#     cv2.imwrite(output_filename, img)
#     print(f"Image saved to {output_filename}")

# print(filtered_classifications_json)
# print(scale_json)

