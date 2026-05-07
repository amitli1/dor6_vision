import cv2
import os
import glob


if __name__ == "__main__":
    # --- CONFIG ---
    for i in range(1,6):
        image_folder = f'/home/amitli/repo/dor6_vision/Testers/tmp_results/VBS_Record_{i}/'
        output_video = f'/home/amitli/repo/dor6_vision/Testers/tmp_results/VBS_Record_{i}/output.mp4'
        frame_duration_ms = 3000                # Duration of each image in milliseconds

        # --- GET IMAGE FILES ---
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort()  # Sort to keep a consistent order

        if not images:
            raise ValueError("No JPG images found in the folder!")

        # --- READ FIRST IMAGE TO GET SIZE ---
        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        # --- VIDEO WRITER SETUP ---
        fps = 1000 / frame_duration_ms  # Frames per second
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # --- ADD IMAGES TO VIDEO ---
        for image_name in images:
            image_path = os.path.join(image_folder, image_name)
            frame = cv2.imread(image_path)
            video.write(frame)

        # --- RELEASE VIDEO ---
        video.release()
        print(f"Video saved as {output_video}")