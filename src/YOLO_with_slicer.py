"""
This module, `YOLO_with_slicer`, is an  wrapper around the YOLOv10 object detection model. 
It introduces enhanced prediction capabilities through image slicing, pre-processing, and various prediction strategies 
to improve detection accuracy, especially in challenging scenarios like detecting small, overlapping.
"""

import torch
from torchvision.ops import nms
from ultralytics import YOLO
import cv2
import numpy as np
from typing import TypeAlias

Slices_list: TypeAlias = list[np.ndarray]
Prediction_result : TypeAlias = list[tuple[torch.Tensor]]


class YOLO_with_slicer:
    """
    YOLO_with_slicer class wraps around a YOLOv10 model to provide enhanced prediction 
    using image slicing techniques to improve object detection performance.
    Notion details noted : 
            YOLO DETECTION : https://prairie-geography-73b.notion.site/YOLO-Training-and-Evaluation-dea277cab338492a8aaeb167ae8d8ed7   
            YOLO IMPROVING : https://prairie-geography-73b.notion.site/YOLO-IMPROVING-075bebe700a74166b0197ee88f2646a1?pvs=4
    Args:
        weight_path (str): Path to the YOLOv10 model weights.
        slicing_overlap (float): Overlap ratio between slices.
        iou_nms (float): IOU threshold for Non-Maximum Suppression (NMS) when merging boundding box and sliced images.
        pre_process (bool) : pre-process by cropping inner bounding box 
        method (str): Prediction method, can be 
            "normal",               ->  Normal prediction
            "slicing",              ->  prediction with slicing inference slicing
            "slicing + normal".     ->  merging predict on both method
            "black_filling"         ->  filling black color in bounding box before slicing
    """

    def __init__(
            self, weight_path:str, 
            slicing_overlap:float = 0.2, 
            iou_nms: float = 0.3, 
            pre_process : bool = False,
            method:str = "slicing + normal") -> None:
        
        self.base_YOLO_model = YOLO(weight_path)
        self.names = self.base_YOLO_model.names
        self.method = method
        self.pre_process = pre_process
        self._overlap = slicing_overlap
        self.iou = iou_nms
        self.crop_origin = (0, 0)  # Default crop origin for pre-processing

    def _pre_process(self,image: np.ndarray) -> np.ndarray:
        """
        pre-processes with cropping inner bounding box for enhance performance of models

        Args:
            image (np.ndarray): images from cv2.imread format RGB

        Returns:
            image (np.ndarray): cropped inner bounding box image
        """
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img
        else:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            # Get the bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            x1, y1 = x , y
            x2, y2 = x + w, y + h
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            self.crop_origin = (x, y)
            return img[y1:y2, x1:x2] 

    
    def _post_process(self, classes:torch.Tensor, scores:torch.Tensor, boxes:torch.Tensor) -> list:
        """
        Post-processes the prediction results to return a more readable format.

        Args:
            classes (Tensor): Tensor containing the class indices of the detected objects.
            scores (Tensor): Tensor containing the confidence scores of the detections.
            boxes (Tensor): Tensor containing the bounding boxes of the detections.

        Returns:
            list: List of detections where each detection is a list containing class index,
                  confidence score, and bounding box coordinates.
        """
        
        return [[cls.item(), score.item(), box.cpu().numpy()] for cls, score, box in zip(classes, scores, boxes)]
    

    def slice_image(self, img:np.ndarray) -> Slices_list:
        """
        Slices the image into tiled based on the given overlap ratio.

        Args:
            img (ndarray): The input image to be sliced.

        Returns:
            list: List of tuples, each containing the coordinates (x, y, x_end, y_end) of the image slices.
        """
        height, width = img.shape[:2]
        slice_size = min(height, width)

        if height > width: # perform Vertical slide
            step_height, step_width = (int(slice_size * (1 - self._overlap)), slice_size)  
        else: # perform Horizontal slide
            step_height, step_width = (slice_size, int(slice_size * (1 - self._overlap)))

        slices = []
        for y in range(0, height, step_height):
            for x in range(0, width, step_width):
                x_end = min(x + slice_size, width)
                y_end = min(y + slice_size, height)
                slices.append((x, y, x_end, y_end))
        
        return slices

    def predict_slices(self, img:np.ndarray, **kwargs)-> Prediction_result:
        """
        Performs prediction on each slice of the image and adjusts the coordinates of the detections.

        Args:
            img (ndarray): The input image to be sliced and predicted on.
            **kwargs: Additional arguments for the base YOLO model's prediction.

        Returns:
            list: List of tuples containing class indices, confidence scores, and bounding boxes for each slice.
                - class indices (Tensor) : predicted classes in each classes
                - confidence scores (Tensor) : confidance scores in each object
                - bounding boxes (Tensor) : bounding boxes in each object in format xyxyn
        """
        all_detections = []
        slices = self.slice_image(img)
        for x, y, x_end, y_end in slices:
            slice_img = img[y:y_end, x:x_end] # slicing images
            results = self.base_YOLO_model(slice_img, **kwargs)[0]
            boxes = results.boxes.xyxy.cpu()
            scores = results.boxes.conf.cpu()
            classes = results.boxes.cls.cpu()

            clone_boxes = boxes.detach().clone()

            clone_boxes[:, [0, 2]] += x 
            clone_boxes[:, [1, 3]] += y
            clone_boxes[:, [0, 2]] /= img.shape[1] #normalize
            clone_boxes[:, [1, 3]] /= img.shape[0] #normalize

            all_detections.append((classes, scores, clone_boxes))
        return all_detections

    def normal_predict(self, img :np.ndarray, **kwargs) -> Prediction_result:
        """
        Performs prediction on the entire image without slicing (Normal slicing).

        Args:
            img (ndarray): The input image to be predicted on.
            **kwargs: Additional arguments for the YOLO model's prediction method.

        Returns:
            list: List of tuples containing class indices, confidence scores, and bounding boxes for each slice.
                - class indices (Tensor) : predicted classes in each classes
                - confidence scores (Tensor) : confidance scores in each object
                - bounding boxes (Tensor) : bounding boxes in each object in format xyxyn
        """
        results = self.base_YOLO_model(img, **kwargs)[0]
        boxes = results.boxes.xyxyn.cpu()
        scores = results.boxes.conf.cpu()
        classes = results.boxes.cls.cpu()
        return [(classes, scores, boxes)]


    def merge_detections(self, detections:list, iou_threshold:float = 0.3) -> list:
        """
        Merges detections from different slices using Non-Maximum Suppression (NMS).

        Args:
            detections (list): List of tuples containing class indices, confidence scores, and bounding boxes for each slice.
            iou_threshold (float): IOU threshold for NMS.

        Returns:
            list: List of tuples containing class indices, confidence scores, 
                    and bounding boxes for each slice. afer eliminate redundant boundding box.
                - class indices (Tensor) : predicted classes in each classes
                - confidence scores (Tensor) : confidance scores in each object
                - bounding boxes (Tensor) : bounding boxes in each object in format xyxyn
        """
        all_boxes, all_scores, all_classes = [], [], []
        for classes, scores, boxes in detections:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        classes = torch.cat(all_classes)
        keep = nms(boxes, scores, iou_threshold= iou_threshold)
        classes = classes[keep].clone().detach()
        scores = scores[keep].clone().detach()
        boxes = boxes[keep].clone().detach()
        return [classes, scores, boxes]

    
    def black_filling(self, img: np.ndarray, detections: list):
        """
        Fills the bounding boxes in the image with black color, where bounding box coordinates are in normalized form.

        Parameters:
        - img (np.ndarray): The input image in which the bounding boxes will be filled.
        - detections (list): A list of tuples, where each tuple contains the class, score, and bounding box.
                             Format: [(class, score, (x1, y1, x2, y2)), ...] with normalized coordinates.

        Returns:
        - img (np.ndarray): The image with bounding boxes filled with black.
        """
        img_height, img_width = img.shape[:2]
        _box_detections = detections[0][2]
        for box in _box_detections:
            x1_n, y1_n, x2_n, y2_n = box
            # Convert normalized coordinates to absolute pixel values
            x1 = int(x1_n * img_width)
            y1 = int(y1_n * img_height)
            x2 = int(x2_n * img_width)
            y2 = int(y2_n * img_height)
            # Ensure the bounding box coordinates are within the image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Fill the bounding box area with black color
            img[y1:y2, x1:x2] = 0
        
        return img
    

    def predict(self, img_path : str, **kwargs) -> list:
        """
        Performs prediction on an image using the specified method.

        Args:
            img_path (str): The file path to the image to be predicted on.
            **kwargs: Additional arguments for the base YOLO model's prediction.

        Returns:
            list: List of detections where each detection is a list containing class index,
                  confidence score, and bounding box coordinates.
        
        Example output: 
            [  #classes(float)  confidence(float)               bounding box in format xyxyn(np.ndarray) 
                [ 0.0,              0.94,             array([    0.20435,     0.33209,     0.21523,     0.37638 ]) ],
                [ 5.0,              0.92,             array([    0.87853,     0.76821,     0.88933,     0.85846 ]) ],
                [ 5.0,              0.62,             array([    0.34586,     0.66268,     0.35692,     0.74187 ]) ],
                .,
                .,
                ...
            ]
        """
        img = cv2.imread(img_path)
        img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.pre_process:
            original_image_shape = img.shape
            img = self._pre_process(img)
            x_offset, y_offset = self.crop_origin
        else:
            x_offset, y_offset = 0, 0

        if self.method == "slicing":
            detections           : Prediction_result = self.predict_slices(img, **kwargs)

        elif self.method == "slicing + normal":
            detections_slicing   : Prediction_result = self.predict_slices(img, **kwargs)
            detections_normal    : Prediction_result = self.normal_predict(img, **kwargs)
            detections           : Prediction_result = detections_slicing + detections_normal # merging detections

        elif self.method == "black_filling" :
            detections          : Prediction_result = self.normal_predict(img, **kwargs)
            _img_black_filling  : np.ndarray        = self.black_filling(img, detections)
            detections_slicing  : Prediction_result = self.predict_slices(_img_black_filling, **kwargs)
            detections          : Prediction_result = detections_slicing + detections       # merging detections

        else:
            detections          : Prediction_result = self.normal_predict(img, **kwargs)

        classes, scores, boxes = self.merge_detections(detections, iou_threshold=self.iou)

        # Adjust the bounding boxes back to the original image coordinates
        if self.pre_process:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * img.shape[1] + x_offset) / original_image_shape[1]
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * img.shape[0] + y_offset) / original_image_shape[0]

        return self._post_process(classes, scores, boxes)