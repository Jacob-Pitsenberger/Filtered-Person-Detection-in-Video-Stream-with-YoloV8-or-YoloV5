"""
Author: Jacob Pitsenberger
Program: people_detector.py
Version: 1.0
Project: Filtered Person Detection in Video Stream with YoloV8 or YoloV5
Date: 8/1/2023
Purpose: This module contains the PeopleDetector class, which is responsible for detecting and counting people
         in a live video feed using YOLO (You Only Look Once) models. It implements the necessary functions to
         process video frames, apply Non-Maximum Suppression (NMS) to filter out duplicate detections, and draw
         bounding boxes around detected people with confidence scores on the video feed. The class can be
         initialized with different YOLO model versions (v5 or v8) to allow flexibility in choosing the
         desired model for the realtime people detection system.
Uses: N/A
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

def non_max_suppression(boxes: np.ndarray, confidences: np.ndarray, threshold: float = 0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter out duplicate detections.

    Parameters:
        boxes (numpy.ndarray): A numpy array containing bounding boxes in [x1, y1, x2, y2] format.
        confidences (numpy.ndarray): A numpy array containing confidence scores for each bounding box.
        threshold (float): The IoU (Intersection over Union) threshold above which duplicate detections will be removed.

    Returns:
        keep (list): A list containing the indices of boxes to keep after NMS.

    """
    # Convert boxes to [x1, y1, x2, y2] format for NMS
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort boxes by their confidence scores in descending order
    order = confidences.argsort()[::-1]

    keep = []  # List to store the indices of boxes to keep after NMS

    while order.size > 0:
        # Keep the box with the highest confidence score
        best_box_idx = order[0]
        keep.append(best_box_idx)

        # Calculate IoU with the other boxes
        xx1 = np.maximum(x1[best_box_idx], x1[order[1:]])
        yy1 = np.maximum(y1[best_box_idx], y1[order[1:]])
        xx2 = np.minimum(x2[best_box_idx], x2[order[1:]])
        yy2 = np.minimum(y2[best_box_idx], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)

        intersection = width * height

        # Calculate IoU
        iou = intersection / (area[best_box_idx] + area[order[1:]] - intersection)

        # Remove boxes with IoU greater than the threshold
        overlapping_boxes = np.where(iou > threshold)[0]
        order = order[overlapping_boxes + 1]

    return keep


class PeopleDetector:
    """
    A class for detecting and counting people in a live video feed using YOLO models.

    Attributes:
        FONT_SIZE (float): Font size for text display on the output window.
        TEXT_COLOR (tuple): RGB color tuple for text display on the output window.
        INFO_BOX_COLOR (tuple): RGB color tuple for drawing the info box on the output window.
        TEXT_THICKNESS (int): Thickness of text display on the output window.
    """

    # Set the confidence threshold (adjust as needed)
    CONFIDENCE_THRESHOLD = 0.5

    FONT_SIZE = 0.5
    TEXT_COLOR = (0, 255, 255)
    INFO_BOX_COLOR = (138, 72, 48)
    TEXT_THICKNESS = 1

    COUNT_LABEL_POSITION = (15, 22)
    COUNT_TEXT_POSITION = (100, 22)

    def __init__(self, model_version: str) -> None:
        """
        Initialize the PeopleDetector object.

        Parameters:
            model_version (str): The version of the YOLO model to use ('v5' or 'v8').
        """
        self.count = 0
        self.model_version = model_version
        self.cap = cv2.VideoCapture(0)
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the YOLO model based on the specified version.
        """
        if self.model_version == "v8":
            self.model = YOLO("yolov8n.pt")
        elif self.model_version == "v5":
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def display_count_box(self, image: np.ndarray) -> None:
        """
        Draws an info box on the output window to display current people count.

        Args:
            image (numpy.ndarray): Input image in BGR format.
        """
        cv2.rectangle(image, (0, 0), (130, 35), self.INFO_BOX_COLOR, -1)
        cv2.putText(image, 'People: ', self.COUNT_LABEL_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        cv2.putText(image, str(self.count), self.COUNT_TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)

    def process_with_v5(self, results: torch.Tensor, frame: np.ndarray):
        """
        Process video frames using YOLOv5 model.

        Args:
            results (torch.Tensor): Results from the YOLOv5 model.
            frame (numpy.ndarray): Current video frame in BGR format.

        Returns:
            frame (numpy.ndarray): Processed video frame with bounding boxes drawn around people.
        """
        # Extract bounding boxes, confidences, and class labels from YOLOv5 detections
        boxes = results.pred[0][:, :4].cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
        confidences = results.pred[0][:, 4].cpu().numpy()  # Confidence scores
        class_labels = results.pred[0][:, 5].cpu().numpy()  # Class labels

        # Filter for the "human" class (class label 0 for YOLOv5, adjust if necessary)
        human_indices = np.where(class_labels == 0)[0]
        human_boxes = boxes[human_indices]
        human_confidences = confidences[human_indices]

        # Apply Non-Maximum Suppression to filter out duplicate detections
        nms_indices = non_max_suppression(human_boxes, human_confidences, threshold=self.CONFIDENCE_THRESHOLD)
        # Create a list to hold people detected.
        people_count = []
        # Draw bounding boxes and add class names with confidence scores for detections above the threshold
        for idx in nms_indices:
            # If detected append the id to the people count list
            people_count.append(idx)
            confidence = human_confidences[idx]
            # Check if the confidence score is above the threshold
            if confidence >= self.CONFIDENCE_THRESHOLD:
                box = human_boxes[idx]

                # Convert box coordinates to integers for drawing
                x1, y1, x2, y2 = box.astype(np.int_)

                # Draw bounding boxes around the human detections on the frame
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'person - {confidence:.2f}'
                frame = cv2.putText(frame, text, (x1, y1 - 10),  cv2.FONT_HERSHEY_COMPLEX, self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        self.count = len(people_count)
        return frame

    def process_with_v8(self, results: list, frame: np.ndarray):
        """
        Process video frames using YOLOv8 model.

        Args:
            results (list): Results from the YOLOv8 model.
            frame (numpy.ndarray): Current video frame in BGR format.

        Returns:
            frame (numpy.ndarray): Processed video frame with bounding boxes drawn around people.
        """

        # Create a list to hold people detected.
        people_count = []
        # Unwrap the detections so that we can get them in our desired format.
        # Only one object so only one iteration so can also call results[0] to get same object.
        for result in results:
            # Iterate through the object treating it as a list.
            for r in result.boxes.data.tolist():
                # Unwrap each class detected as its id, bounding box coordinates, and the confidence score for detecting.
                x1, y1, x2, y2, score, class_id = r
                # Get the bounding box coordinates and class id as integers
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)

                # Filter for the "person" class (class id 0 for YOLOv5, adjust if necessary)
                if class_id == 0:
                    # Apply Non-Maximum Suppression to filter out duplicate detections
                    boxes = np.array([[x1, y1, x2, y2]])
                    confidences = np.array([score])
                    nms_indices = non_max_suppression(boxes, confidences)

                    # Draw bounding boxes around the human detections on the frame
                    for idx in nms_indices:
                        # If detected append the id to the people count list
                        people_count.append(idx)

                        confidence = confidences[idx]

                        # Set the confidence threshold (adjust as needed)
                        confidence_threshold = 0.5

                        if confidence >= confidence_threshold:
                            box = boxes[idx]

                            x1, y1, x2, y2 = box.astype(np.int_)
                            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f'person - {confidence:.2f}'
                            frame = cv2.putText(frame, text, (x1, y1 - 10),  cv2.FONT_HERSHEY_COMPLEX, self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        self.count = len(people_count)
        return frame

    def process_video(self):
        """
        Process the live video feed and display the output.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            # Make detections using the YOLOv5 model on the current frame
            results = self.model(frame)

            if self.model_version == "v5":
                frame = self.process_with_v5(results, frame)
            elif self.model_version == "v8":
                frame = self.process_with_v8(results, frame)

            self.display_count_box(frame)
            cv2.imshow('output', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self) -> None:
        """
        Releases the video capture when the object is deleted.
        """
        if self.cap.isOpened():
            self.cap.release()
