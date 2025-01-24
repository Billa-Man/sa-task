import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from PIL import Image


def hand_points_and_labels(rgb_image_filepath):

    rgb_image = Image.open(rgb_image_filepath)
    width, height = rgb_image.size

    base_options = python.BaseOptions(model_asset_path='models/mediapipe/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(rgb_image_filepath)
    detection_result = detector.detect(image)

    left_hand_pixels = []
    right_hand_pixels = []

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        pixels = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks]
        handedness_label = detection_result.handedness[idx][0].category_name
        
        if handedness_label == "Left":
            left_hand_pixels.append(pixels)
        elif handedness_label == "Right":
            right_hand_pixels.append(pixels)

    print(left_hand_pixels)
    print(right_hand_pixels)

    return np.array(left_hand_pixels, dtype=np.uint8), np.array(right_hand_pixels, dtype=np.uint8)


if __name__ == "__main__":

    image_path = "media/test_frames/00000.jpg"
    hand_points_and_labels(image_path)



