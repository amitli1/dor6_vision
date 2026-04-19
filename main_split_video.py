import pandas as pd
import cv2
import os
import glob

INPUT_VIDEO_FILE = "./Dataset/video/Train_A_Video.mp4"

def print_video_statiscs(mp4_file_path):
    video        = cv2.VideoCapture(mp4_file_path)
    frame_count  = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps          = video.get(cv2.CAP_PROP_FPS)
    video_length = frame_count / fps

    print(f"Video Length           : {video_length:.2f} seconds ({(video_length/60):.2f} minutes)")
    print(f"Frames per Second (FPS): {fps}")

def split_video_to_jpg_files(mp4_file_path, output_folder):

    cap = cv2.VideoCapture(mp4_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval to capture one frame per second
    frame_interval = int(fps)  # Integer value of FPS to capture 1 frame per second

    frame_count       = 0
    saved_frame_count = 0

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Break if no frame is read (end of video)
        if not ret:
            break

        # Check if this is the correct frame to save (every 1 second)
        if frame_count % frame_interval == 0:
            # Save the frame as JPEG
            output_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_file, frame)
            print(f"Saved {output_file}")
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()

def create_jpg_dataframe():
    l_files = glob.glob("./Dataset/test_set/*.jpg")
    l_base  = []
    for file in l_files:
        l_base.append(os.path.basename(file))
    l_base.sort()
    df = pd.DataFrame({"jpg_file" : l_base})
    df.to_csv("/home/amitli/repo/dor6_vision/Dataset/test_set.csv", index=False)


if __name__ == "__main__":
    print_video_statiscs(INPUT_VIDEO_FILE)
    split_video_to_jpg_files(INPUT_VIDEO_FILE, "./Dataset/test_set")
    create_jpg_dataframe()

    None

