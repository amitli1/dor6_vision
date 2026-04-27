import base64
from openai import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import json

from app_config.settings import TRAIN_FULL_MODE_FILES_PATH


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def img_to_content(path):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(path)}"
        },
    }


def get_point_prompt(target_img_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Detect the objects in this image and provide a single center point for each in $[y, x]$ format. Use normalized coordinates from 0 to 1000. For example, an object in the dead center of the image should be labeled as [500, 500]."},
                img_to_content(target_img_path)
            ],
        }
    ]
    return messages


def get_bounding_box_prompt(target_img_path):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Detect the military vehicle in this image and provide their bounding boxes in [ymin, xmin, ymax, xmax] format."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(target_img_path)}"
                    },
                },
            ],
        }
    ]
    return messages


if __name__ == "__main__":
    sa_22_path_1 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-21-02_1244400_1020.jpg"
    sa_22_path_2 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-20-27_844400_795.jpg"
    sa_22_path_3 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-17-44_444400_23.jpg"

    scud_path_1 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-21-10_1324400_673.jpg"
    scud_path_2 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-17-54_524400_136.jpg"
    scud_path_3 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-20-34_884400_1224.jpg"

    t_90_path_1 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-20-40_924400_731.jpg"
    t_90_path_2 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-18-04_484400_633.jpg"
    t_90_path_3 = f"{TRAIN_FULL_MODE_FILES_PATH}/11-21-17_1284400_311.jpg"

    l_files = [sa_22_path_1 ,sa_22_path_2 , sa_22_path_3 ,
               scud_path_1 , scud_path_2 , scud_path_3 ,
               t_90_path_1 ,t_90_path_2 , t_90_path_3]

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    for full_path_file in l_files:

        base_file = os.path.basename(full_path_file)
        messages = get_bounding_box_prompt(full_path_file)
        response = client.chat.completions.create(
            model="google/gemma-4-31B-it",
            messages=messages,
        )
        res_text = response.choices[0].message.content
        print(f"[{base_file}] result: {res_text}")

    print("\n")
    for full_path_file in l_files:
        base_file = os.path.basename(full_path_file)
        messages = get_point_prompt(full_path_file)
        response = client.chat.completions.create(
            model="google/gemma-4-31B-it",
            # model="/model_path",
            messages=messages,
            max_tokens=500
        )
        res_text = response.choices[0].message.content
        print(f"[{base_file}] result: {res_text}")



# /home/amitli/repo/dor6_vision/.venv/bin/python /home/amitli/repo/dor6_vision/main_get_bouding_box.py
# [11-21-02_1244400_1020.jpg] result: ```json
# [
#   {"box_2d": [357, 398, 437, 476], "label": "military vehicle"}
# ]
# ```
# [11-20-27_844400_795.jpg] result: ```json
# [
#   {"box_2d": [544, 479, 650, 536], "label": "military vehicle"}
# ]
# ```
# [11-17-44_444400_23.jpg] result: ```json
# [
#   {"box_2d": [472, 352, 821, 538], "label": "military vehicle"}
# ]
# ```
# [11-21-10_1324400_673.jpg] result: ```json
# [
#   {"box_2d": [487, 532, 555, 627], "label": "military vehicle"}
# ]
# ```
# [11-17-54_524400_136.jpg] result: ```json
# [
#   {"box_2d": [207, 436, 392, 630], "label": "military vehicle"}
# ]
# ```
# [11-20-34_884400_1224.jpg] result: ```json
# [
#   {"box_2d": [377, 494, 462, 548], "label": "military vehicle"}
# ]
# ```
# [11-20-40_924400_731.jpg] result: ```json
# [
#   {"box_2d": [485, 566, 551, 615], "label": "military vehicle"}
# ]
# ```
# [11-18-04_484400_633.jpg] result: ```json
# [
#   {"box_2d": [315, 450, 467, 515], "label": "military vehicle"}
# ]
# ```
# [11-21-17_1284400_311.jpg] result: ```json
# [
#   {"box_2d": [254, 320, 456, 465], "label": "military vehicle"}
# ]
# ```
#
#
# [11-21-02_1244400_1020.jpg] result: ```json
# [
#   {"point": [395, 437]}
# ]
# ```
# [11-20-27_844400_795.jpg] result: ```json
# [
#   {"point": [595, 506], "label": "boat"}
# ]
# ```
# [11-17-44_444400_23.jpg] result: ```json
# [
#   {"point": [641, 444], "label": "objects"}
# ]
# ```
# [11-21-10_1324400_673.jpg] result: ```json
# [
#   {"point": [525, 578], "label": "objects"}
# ]
# ```
# [11-17-54_524400_136.jpg] result: ```json
# [
#   {"point": [312, 532], "label": "the objects"}
# ]
# ```
# [11-20-34_884400_1224.jpg] result: ```json
# [
#   {"point": [424, 524], "label": "object"}
# ]
# ```
# [11-20-40_924400_731.jpg] result: I did not find any bounding box detections for .
# [11-18-04_484400_633.jpg] result: ```json
# [
#   {"point": [411, 486], "label": "tank"}
# ]
# ```
# [11-21-17_1284400_311.jpg] result: ```json
# [
#   {"point": [362, 408], "label": "tank"}
# ]
# ```
#
# Process finished with exit code 0