import base64
import cv2
from openai import OpenAI
import time

def encode_image_from_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def process_video_and_detect(video_path):
    # 2. Extract Frames (Sampling at 1 FPS)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []

    count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        # Sample one frame per second
        if count % int(fps) == 0:
            base64_frame = encode_image_from_frame(frame)
            frames.append(base64_frame)
        count += 1
    video.release()

    # 3. Construct the Multimodal Message
    # We pass the frames as a list of image_url objects
    content = [
        {
            "type": "text",
            "text": (
                "Detect all military vehicles in this simulation flight shot. "
                "For every detected vehicle, provide the normalized bounding box [ymin, xmin, ymax, xmax] "
                "and its classification (e.g., Main Battle Tank, APC, Mobile Artillery, etc.). "
                "Output as a structured list."
            )
        }
    ]

    for base64_image in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # 4. API Call with Gemma 4 Specific Parameters
    response = client.chat.completions.create(
        model="google/gemma-4-31b-it",
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
        # Pass visual budget and thinking mode in extra_body for local engines supporting it
        extra_body={
            "visual_token_budget": 1120,
            "enable_thinking": True
        }
    )

    return response.choices[0].message.content



def annotate_video(input_file, output_file, detection_data):
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    detections = detection_data["detections"]

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current timestamp of the frame in milliseconds
        current_ms = int((frame_idx / fps) * 1000)

        # Draw boxes if the timestamp matches (with a 500ms visibility window)
        for det in detections:
            # We show the box if the frame is within 500ms of the detection timestamp
            if 0 <= (current_ms - det["timestamp_ms"]) < 500:
                ymin, xmin, ymax, xmax = det["box_2d"]

                # Convert normalized (0-1000) to pixel coordinates
                # Note: Gemma coordinates are [ymin, xmin, ymax, xmax]
                left = int(xmin * width / 1000)
                top = int(ymin * height / 1000)
                right = int(xmax * width / 1000)
                bottom = int(ymax * height / 1000)

                # Draw the Red Rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw the Label with a small black background for readability
                label = det["label"]
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved as {output_file}")

if __name__ == "__main__":
    response = {"detections": [{"timestamp_ms": 8000, "box_2d": [838, 137, 880, 167], "label": "military vehicle"},
                               {"timestamp_ms": 9000, "box_2d": [416, 354, 445, 378], "label": "military vehicle"},
                               {"timestamp_ms": 9000, "box_2d": [775, 274, 814, 302], "label": "military vehicle"},
                               {"timestamp_ms": 10000, "box_2d": [767, 282, 807, 312], "label": "military vehicle"},
                               {"timestamp_ms": 11000, "box_2d": [952, 232, 995, 261], "label": "military vehicle"},
                               {"timestamp_ms": 12000, "box_2d": [844, 287, 887, 318], "label": "military vehicle"},
                               {"timestamp_ms": 12000, "box_2d": [941, 223, 988, 255], "label": "military vehicle"},
                               {"timestamp_ms": 13000, "box_2d": [837, 288, 880, 318], "label": "military vehicle"},
                               {"timestamp_ms": 14000, "box_2d": [874, 242, 924, 280], "label": "military vehicle"},
                               {"timestamp_ms": 14000, "box_2d": [952, 231, 998, 261], "label": "military vehicle"},
                               {"timestamp_ms": 15000, "box_2d": [945, 234, 991, 265], "label": "military vehicle"},
                               {"timestamp_ms": 16000, "box_2d": [945, 232, 991, 264], "label": "military vehicle"},
                               {"timestamp_ms": 16000, "box_2d": [807, 104, 857, 161], "label": "military vehicle"}]}

    FILE = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/videos/VBS_Record_5.mp4'
    annotate_video(FILE, '/home/amitli/repo/dor6_vision/results/videos/out.mp4', response)
    exit(0)


    # 1. Initialize Client
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    FILE = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/videos/VBS_Record_5.mp4'
    #result = process_video_and_detect("1.mp4")
    #print(result)
    with open(FILE, "rb") as video_file:
        base64_video = base64.b64encode(video_file.read()).decode("utf-8")

    # 2. Call the model
    # The model will automatically sample 1 frame per second from the file
    SCHEMA = {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp_ms": {"type": "integer", "description": "Time in milliseconds"},
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "[ymin, xmin, ymax, xmax] normalized 0-1000"
                        },
                        "label": {"type": "string",
                                  "description": "Specific vehicle type (e.g., M1 Abrams, T-90, Stryker)"}
                    },
                    "required": ["timestamp_ms", "box_2d", "label"]
                }
            }
        },
        "required": ["detections"]
    }

    prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Watch this simulation flight shot. Detect all military vehicles "
                            "and provide their bounding boxes [ymin, xmin, ymax, xmax] "
                            "and specific classifications. Format as JSON."
                        )
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"}
                    }
                ]
            }
        ]

    start_time = time.time()
    response = client.chat.completions.create(
        model = "google/gemma-4-31B-it",
        messages=prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "military_analysis", "schema":SCHEMA}
        },
        extra_body={
            "visual_token_budget": 1120,  # High detail for vehicle classification
            "enable_thinking": True  # Reasoning for specific model variants
        }
    )
    elapse_time = time.time() - start_time
    print(f'{elapse_time} seconds')

    print(response.choices[0].message.content)



