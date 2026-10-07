
import cv2
import os

video_path = "data/raw/Smash/smash1.mp4"
output_folder = "data/processed/Smash/smash1"

os.makedirs(output_folder, exist_ok=True)

video = cv2.VideoCapture(video_path)

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", total_frames)

num_frames = 16

for i in range(num_frames):

    frame_position = int(i * total_frames / num_frames)

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

    success, frame = video.read()

    if success:

        frame = cv2.resize(frame, (224, 224))

        output_path = os.path.join(
            output_folder,
            f"frame_{i:02d}.jpg"
        )

        cv2.imwrite(output_path, frame)

        print("Saved:", output_path)

video.release()

print("Frame extraction and resizing completed!")
