import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import numpy as np
import cv2
from PIL import Image

def save_output_video(frame_names, video_segments, video_dir, output_video_filepath):

    first_frame_path = os.path.join(video_dir, frame_names[0])
    first_frame = Image.open(first_frame_path)
    frame_width, frame_height = first_frame.size

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(output_video_filepath, fourcc, fps, (frame_width, frame_height))
    
    # Create color map for different object IDs (red and blue)
    color_map = {
        0: [255, 0, 0],  # Red
        1: [0, 0, 255]   # Blue
    }
    
    for frame_idx in range(len(frame_names)):

        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        frame = np.array(Image.open(frame_path))

        # Overlay segmentation masks
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                color = color_map[obj_id % 2]
                
                mask = mask.astype(np.uint8) * 255
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :, 0] = mask * color[2]  # Blue
                mask_colored[:, :, 1] = mask * color[1]  # Green
                mask_colored[:, :, 2] = mask * color[0]  # Red
                frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.5, 0)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
