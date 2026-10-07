import cv2
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

CLASSES = ["clear", "drive", "drop", "net", "smash"]
NUM_FRAMES = 16

for class_name in CLASSES:

    input_folder = os.path.join(RAW_DIR, class_name)
    output_folder = os.path.join(PROCESSED_DIR, class_name)

    os.makedirs(output_folder, exist_ok=True)

    for video_name in os.listdir(input_folder):

        video_path = os.path.join(input_folder, video_name)

        if not video_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video = cv2.VideoCapture(video_path)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video_id = os.path.splitext(video_name)[0]

        video_output = os.path.join(output_folder, video_id)

        os.makedirs(video_output, exist_ok=True)

        for i in range(NUM_FRAMES):

            frame_number = int(i * total_frames / NUM_FRAMES)

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            success, frame = video.read()

            if success:
                frame_path = os.path.join(
                    video_output,
                    f"frame_{i+1:02d}.jpg"
                )

                cv2.imwrite(frame_path, frame)

        video.release()

print("Frame extraction completed!")
