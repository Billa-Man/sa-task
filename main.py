import os

from config import Config
config = Config()

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

from utils.sam2_utils.create_video_dir import extract_frames
from utils.mediapipe.points_and_labels import hand_points_and_labels

import cv2

#---------- SET DEVICE ----------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

#---------- LOAD SAM2 MODEL ----------
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

#---------- LOAD VIDEO FRAME-BY-FRAME ----------
def main_function(input_video_filepath, output_video_filepath):

    extract_frames(input_video_filepath, config.video_frames_dir)

    frame_names = [
        p for p in os.listdir(config.video_frames_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Initialize tracking
    inference_state = predictor.init_state(video_path=config.video_frames_dir)
    predictor.reset_state(inference_state)
    
    # Process first frame for hand detection
    first_frame_path = os.path.join(config.video_frames_dir, frame_names[0])
    left_hand_pixels, right_hand_pixels = hand_points_and_labels(first_frame_path)
    left_hand_pixels = np.squeeze(left_hand_pixels)
    right_hand_pixels = np.squeeze(right_hand_pixels)

    # Set up hand tracking
    ann_left_hand_obj_id, ann_right_hand_obj_id = 1, 2
    
    # Create labels for each hand
    labels_left = np.ones(left_hand_pixels.shape[0], dtype=np.uint8)
    labels_right = np.ones(right_hand_pixels.shape[0], dtype=np.uint8)
    
    # Process hands
    for hand_pixels, hand_id, labels in [
        (left_hand_pixels, ann_left_hand_obj_id, labels_left),
        (right_hand_pixels, ann_right_hand_obj_id, labels_right)
    ]:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=hand_id,
            points=hand_pixels,
            labels=labels
        )

    output_frames_dir = "./output/processed_frames/"
    os.makedirs(output_frames_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width, frame_height = 1280, 720
    output_fps = 30
    video_writer = cv2.VideoWriter(output_video_filepath, fourcc, output_fps, (frame_width, frame_height))

    video_segments = {}  # This contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        frame_path = os.path.join(config.video_frames_dir, frame_names[out_frame_idx])
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (frame_width, frame_height))

        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        for obj_id, mask in video_segments[out_frame_idx].items():

            mask = np.squeeze(mask)
            mask = cv2.resize(mask.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

            color = (0, 255, 0) if obj_id == 1 else (0, 0, 255)  # Green for left hand, Red for right hand
            colored_mask[mask > 0] = color

        alpha = 0.5
        frame = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)

        video_writer.write(frame)

        frame_output_path = os.path.join(output_frames_dir, f"frame_{out_frame_idx:04d}.jpg")
        cv2.imwrite(frame_output_path, frame)

    video_writer.release()
    
#---------- MAIN FUNCTION ----------
if __name__ == "__main__":
    input_video_filepath = "media/test.mp4"
    output_video_filepath = "output/test_output_with_mask.mp4"
    main_function(input_video_filepath, output_video_filepath)


