import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir, quality=100):

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if not cap.isOpened():
        return False
    
    frame_count = 0
    with tqdm(total=total_frames, desc="Extracting Frames", 
            unit="frames", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            output_path = os.path.join(output_dir, f'{frame_count:05d}.jpg')
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            frame_count += 1
            pbar.update(1)
            pbar.set_postfix({"FPS": fps})
    
    cap.release()
    return True


if __name__ == "__main__":
    video_path = "media/test.mp4"
    output_dir = "media/test_frames"
    extract_frames(video_path, output_dir)