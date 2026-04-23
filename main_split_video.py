import pandas as pd
import cv2
import os
import glob

from app_config.settings import VIDEO_TEST_FILE_PATH, VIDEO_TEST_PREDICTION

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

def create_video_prediction():

    cap = cv2.VideoCapture(VIDEO_TEST_FILE_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)  # every 0.5 seconds

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_TEST_PREDICTION, fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # every 0.5 sec frame
        if frame_idx % frame_interval == 0:
            # draw red circle
            center = (width // 2, height // 2)
            cv2.circle(frame, center, 50, (0, 0, 255), 3)

            # add text
            text = f"t = {frame_idx / fps:.1f}s"
            cv2.putText(frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    print_video_statiscs(INPUT_VIDEO_FILE)
    CREATE_TEST_FRAMES        = False
    CREATE_MP4_WITH_PREDICTION = True

    if CREATE_TEST_FRAMES:
        split_video_to_jpg_files(INPUT_VIDEO_FILE, "./Dataset/test_set")
        create_jpg_dataframe()

    if CREATE_MP4_WITH_PREDICTION:
        create_video_prediction()

    None

