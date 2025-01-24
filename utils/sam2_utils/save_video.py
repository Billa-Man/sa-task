import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import numpy as np
import cv2
from PIL import Image

def save_output_video(frame_names, video_segments, video_dir, output_video_filepath):
    # Get dimensions from first frame
    first_frame_path = os.path.join(video_dir, frame_names[0])
    first_frame = Image.open(first_frame_path)
    frame_width, frame_height = first_frame.size

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(output_video_filepath, fourcc, fps, (frame_width, frame_height))

    for frame_idx in range(len(frame_names)):
        # Read frame
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        frame = np.array(Image.open(frame_path))

        # Overlay segmentation masks
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                color = (0, 255, 0)  # Green overlay for masks
                mask = mask.astype(np.uint8) * 255
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :, 1] = mask
                frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.5, 0)

        # Write frame to video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
