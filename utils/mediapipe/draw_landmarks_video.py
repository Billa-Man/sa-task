import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import cv2
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.mediapipe.draw_landmarks_image import draw_landmarks_on_image

from config import Config
config = Config()


def draw_landmarks_on_video(video_filepath, output_path="output/output_video.mp4"):
    try:
        cap = cv2.VideoCapture(video_filepath)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise ValueError("Error creating output video file")

        base_options = python.BaseOptions(model_asset_path="models/mediapipe/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                             num_hands=config.num_hands)
        detector = vision.HandLandmarker.create_from_options(options)

        with tqdm(total=total_frames, desc="Drawing Landmarks on Video", 
                 unit="frames", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = detector.detect(mp_image)
                annotated_frame = draw_landmarks_on_image(frame, detection_result)
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

                pbar.update(1)
                pbar.set_postfix({"FPS": fps})

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    media_filepath = "media/test.mp4"
    output_filepath = "output/output_video_test.mp4"

    draw_landmarks_on_video(media_filepath, output_filepath)

