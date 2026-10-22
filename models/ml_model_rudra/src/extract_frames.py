# import cv2
# video = cv2.videocapture("test.mp4")
# while True:
#     ret,frame=video.read()
#     if not ret:
#         break
#     cv2.imshow("badmention video",frame)
#     if cv2.waitKey(1) == 27:
#         break
#     video.release()
#     cv2.destroyAllWindows()
    
# import cv2
# import os

# video_path = "data/raw/smash/smash1.mp4"
# output_folder = "data/processed"

# os.makedirs(output_folder, exist_ok=True)

# video = cv2.VideoCapture(video_path)

# total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# print("Total frames:", total_frames)

# num_frames = 20


# for i in range(num_frames):

#     frame_number = int(i * total_frames / num_frames)

#     video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

#     ret, frame = video.read()

#     if ret:

#         frame = cv2.resize(frame, (224, 224))

#         filename = os.path.join(
#             output_folder,
#             f"frame_{i:02d}.jpg"
#         )

#         cv2.imwrite(filename, frame)

#         print("Saved:", filename)

# video.release()

# print("Frame extraction completed!")
import cv2
import os


# Main folders
input_folder = "data/raw"
output_folder = "data/processed"

# Number of frames to extract from each video
num_frames = 16


# Go through every class folder
for class_name in os.listdir(input_folder):

    class_path = os.path.join(input_folder, class_name)

    # Skip anything that is not a folder
    if not os.path.isdir(class_path):
        continue

    # Create class output folder
    class_output = os.path.join(output_folder, class_name)
    os.makedirs(class_output, exist_ok=True)


    # Go through every video inside the class
    for video_name in os.listdir(class_path):

        # Only process MP4 files
        if not video_name.lower().endswith((".mp4","avi")):
            continue


        # Full video path
        video_path = os.path.join(class_path, video_name)

        # Remove .mp4 from video name
        video_id = os.path.splitext(video_name)[0]

        # Folder where frames will be saved
        video_output = os.path.join(class_output, video_id)
        os.makedirs(video_output, exist_ok=True)


        print("\nProcessing:", video_path)


        # Open video
        video = cv2.VideoCapture(video_path)

        # Get total number of frames
        total_frames = int(
            video.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        print("Total frames:", total_frames)


        # Check whether video opened successfully
        if total_frames <= 0:
            print("Could not read video:", video_path)
            video.release()
            continue


        # Extract frames
        for i in range(num_frames):

            # Select frame position
            frame_number = int(
                i * total_frames / num_frames
            )

            # Move video to that frame
            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            # Read frame
            ret, frame = video.read()


            # If frame was successfully read
            if ret:

                # Resize to 224 x 224
                frame = cv2.resize(
                    frame,
                    (224, 224)
                )

                # Save frame
                filename = os.path.join(
                    video_output,
                    f"frame_{i:02d}.jpg"
                )

                cv2.imwrite(
                    filename,
                    frame
                )

            else:
                print(
                    "Could not read frame:",
                    frame_number
                )


        # Close video
        video.release()

        print(
            "Completed:",
            video_name
        )


print("\nAll videos processed!")