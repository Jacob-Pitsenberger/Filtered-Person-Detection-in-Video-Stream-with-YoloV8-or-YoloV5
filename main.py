"""
Author: Jacob Pitsenberger
Program: main.py
Version: 1.0
Project: Filtered Person Detection in Video Stream with YoloV8 or YoloV5
Date: 8/1/2023
Purpose: This program contains the main method for running the Realtime People Detection in Video Feed System.
Uses: people_detector.py
"""

from people_detector import PeopleDetector


def main():
    """Main method to run the Realtime People Detection in Video Feed System.

    Returns:
        None
    """
    versions = ["v5", "v8"]

    detector = PeopleDetector(versions[1])
    detector.process_video()


if __name__ == "__main__":
    main()
