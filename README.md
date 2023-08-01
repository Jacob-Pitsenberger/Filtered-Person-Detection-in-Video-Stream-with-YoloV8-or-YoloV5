# Filtered-Person-Detection-in-Video-Stream-with-YoloV8-or-YoloV5

Author: Jacob Pitsenberger
Version: 1.0
Date: 7/27/2023

## Description

This project implements a Realtime People Detection System in a live video feed using YOLO (You Only Look Once) models. The system can detect and count people in real time, making it useful for various applications such as monitoring public places, crowd analysis, and security surveillance.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Ultralytics (for YOLO model)
- PyTorch (for YOLOv5 model)

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/Jacob-Pitsenberger/Filtered-Person-Detection-in-Video-Stream-with-YoloV8-or-YoloV5.git>
cd <Filtered-Person-Detection-in-Video-Stream-with-YoloV8-or-YoloV5>
```

1. Install the required packages:

```bash
pip install opencv-python numpy ultralytics torch
```

## Usage

To run the Realtime People Detection in Video Feed System, use the main.py script:

```bash
python main.py
```
You will be prompted to select the YOLO model version to use (v5 or v8). The system will then open your default camera feed and start detecting and counting people in real time. To stop the program, press 'q'.

## Configuration

You can adjust the confidence threshold for person detection by modifying the CONFIDENCE_THRESHOLD attribute in the PeopleDetector class of people_detector.py.

## Model Versions

YOLOv5: A lightweight version of YOLO that provides real-time inference with good accuracy.
YOLOv8: A more advanced version of YOLO that may offer better performance at the cost of increased complexity and inference time.

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to the developers of the YOLO models and the Ultralytics library for making this project possible.



