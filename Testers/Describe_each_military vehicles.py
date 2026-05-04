from openai                          import OpenAI
from PIL                             import Image, ImageDraw, ImageFont
from app_config.settings             import FONT_FILE, TRAIN_FULL_MODE_FILES_PATH

import base64
import json
import pandas as pd
import os
import glob
import time

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def add_text_line(content, text):
    content.append({"type": "text", "text": f"{text}"})

def add_image_line(content, image_path):
    content.append({"type": "image_url", "image_url": {"url":f"data:image/jpeg;base64,{encode_image(image_path)}"}})

def build_prompt_get_describe_full_image(target_img):
    base_prompt = '''
    You are given a simulation image.
    
    Task:
    1. Detect all visible military vehicles in the image.
    2. For each detected object, return:
       - bounding box
       - class (if identifiable)
       - a concise description of distinguishing visual features useful for classification.
       
    Description requirements:
    - Focus ONLY on structural and geometric features useful for classification:
      shape, proportions, turret presence, barrel length, number of wheels/tracks, layout, mounted equipment.
    - DO NOT mention color, shading, lighting, or texture.
    - Assume all vehicles have identical color; color is not a valid feature.
    
    Definitions:
    - Military vehicles include: tanks, armored personnel carriers (APC), trucks, artillery vehicles, missile launchers.
    - Ignore non-military objects (buildings, people, terrain, etc.).
    
    Bounding box format:
    - Use [x_min, y_min, x_max, y_max] in pixel coordinates relative to the original image.
    
    Description requirements:
    - Focus ONLY on visual features useful for classification, such as:
      shape, size, turret presence, number of wheels/tracks, color patterns, mounted equipment.
    - Avoid generic phrases like "military vehicle" or "object in image".
    
    Uncertainty handling:
    - If an object is partially visible or unclear, still return it but set "uncertain": true.
    - If no military vehicles are present, return an empty list.

    '''
    content = []
    add_text_line(content, base_prompt)
    add_image_line(content, target_img)

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages


def build_prompt_get_describe_the_crop_image(target_img):
    base_prompt = ("Describe the military vehicles in this simulation image so I can identify it later. If the image is very noisy or unclear - write it")
    content = []
    add_text_line(content, base_prompt)
    add_image_line(content, target_img)

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages

def describe_the_crop_image(client, full_image_path):
    messages = build_prompt_get_describe_the_crop_image(full_image_path)
    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        extra_body={
            "mm_processor_kwargs": {
                "max_soft_tokens": 280
            }
        }
    )
    res_text = response.choices[0].message.content
    return res_text


def describe_the_full_image(client, full_image_path):
    messages = build_prompt_get_describe_full_image(full_image_path)
    schema = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "class": {
                            "type": "string",
                            "enum": ["SA-22", "SCUD", "T-90", "unknown"]
                        },
                        "description": {
                            "type": "string"
                        },
                        "uncertain": {
                            "type": "boolean"
                        }
                    },
                    "required": ["bbox", "class", "description", "uncertain"]
                }
            }
        },
        "required": ["objects"]
    }
    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        extra_body={
            "guided_json": schema,
            "mm_processor_kwargs": {
                "max_soft_tokens": 1120
            }
        }
    )
    res_text = response.choices[0].message.content
    res_text = res_text.replace("```json", "").replace("```", "").strip()
    res_json = json.loads(res_text)

    return res_json

def draw_result(image_path, gemma_res, draw_plot=True):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    font = ImageFont.truetype(FONT_FILE, size=32)

    pred_str = ""
    for i, res in enumerate(gemma_res):
        if 'class' in res:
            label       = res['class']
        else:
            label = res['label']
        description = res['description']
        uncertain   = res['uncertain']
        pred_str += f"[{i+1}] [{label}] [{description}] [{uncertain}]" + "\n"

    draw.text((1, 1), pred_str, fill="red", font=font)

    for i, res in enumerate(gemma_res):
        box_2d      = res['bounding_box']
        if 'class' in res:
            label = res['class']
        else:
            label = res['label']

        ymin, xmin, ymax, xmax = box_2d

        left = xmin * width / 1000
        top = ymin * height / 1000
        right = xmax * width / 1000
        bottom = ymax * height / 1000
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        draw.text((left + 50, top), f"{i+1} {label}", fill="red", font=font)

    if draw_plot:
        img.show()

if __name__ == "__main__":
    client    = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    RUN_ON_FULL_IMAGE = True
    RUN_ON_CROP       = False

    if RUN_ON_FULL_IMAGE:
        #image_path  = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_3/frame_273_00_06_418.jpg'
        #image_path  = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_1/frame_266_00_06_853.jpg'
        #image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_1/frame_190_00_04_895.jpg'

        sa_22_1 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-44_444400_23.jpg'
        sa_22_2 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-20-27_844400_795.jpg'
        sa_22_3 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-21-02_1244400_1020.jpg'

        scud_1 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-54_524400_136.jpg'
        scud_2 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-20-34_884400_1224.jpg'
        scud_3 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-21-10_1324400_673.jpg'

        t90_1 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-18-04_484400_633.jpg'
        t90_2 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-20-40_924400_731.jpg'
        t90_3 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-21-17_1284400_311.jpg'

        t90_err_1 = '11-20-40_924400_966.jpg'
        t90_err_2 = '11-21-17_1284400_1482.jpg'

        sa22_err_1 = '11-21-02_1244400_1657.jpg'
        sa22_err_2 = '11-20-27_844400_1549.jpg'
        ''

        FILES_PATH = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/'
        #FILE_NAMES = [t90_err_1, t90_err_2]
        FILE_NAMES  = [sa22_err_1, sa22_err_2]

        # FILES_PATH = '/home/amitli/repo/dor6_vision/Testers/few_shots/'
        # FILE_NAMES = ['SA-22_1.JPG', 'SA-22_2.JPG', 'SA-22_3.JPG',
        #                      'SCUD_1.JPG', 'SCUD_2.JPG', 'SCUD_3.JPG',
        #                      'T-90_1.JPG', 'T-90_2.JPG', 'T-90_3.JPG',
        #                      'OTHER_1.jpg']

        for few_shot in FILE_NAMES:
            image_path = f"{FILES_PATH}{few_shot}"
            start_time  = time.time()
            description = describe_the_full_image(client, image_path)
            end_time    = time.time()
            print("\n-------------------------------------------------------")
            print(json.dumps(description, indent=4))
            print(f"{few_shot}")
            print(f"Elapsed time: {(end_time - start_time):.2f}")
            draw_result(image_path, description, draw_plot=False)

        # SCUD
        # ??? missile launcher
        # long rectangular chassis with rail-mounted launch canisters
        # Multi-axle wheeled chassis with a long, cylindrical missile mounted on the rear deck.
        # "Rectangular chassis, elongated body, lack of turret or visible heavy weaponry, wheeled layout.",
        # "long rectangular chassis with a large cylindrical missile canister mounted on top, multiple wheels",

        # SA-22
        #  lacking a prominent long barrel."

        # ??? missile launcher
        # Multi-axle truck chassis with four visible wheels on one side, equipped with a large rectangular rear platform holding four cylindrical missile canisters oriented vertically/diagonally."
        # Rectangular chassis with a distinct cabin at the front and an open cargo bed at the rear.",
        # Rectangular chassis with a flatbed rear section and a forward-positioned cabin.

        # NO SA-22
        # Rectangular chassis, tan color, multi-axle configuration

        # T-90
        # tank
        # a visible turret profile
        # tracked chassis with a centrally mounted turret and a long forward-facing gun barrel.
        # Rectangular chassis with a centered turret and a short forward-facing barrel
        # Rectangular chassis with a centered turret and a short forward-facing barrel
        # Tracked vehicle with a central rotating turret and a long forward-projecting barre


    if RUN_ON_CROP:
        l_crops   = glob.glob('/home/amitli/repo/dor6_vision/Testers/tmp_files/*.jpg')
        for crop_file in l_crops:

            base_name = os.path.basename(crop_file)
            if base_name != 'crop_4.jpg':
                continue

            description = describe_the_crop_image(client, crop_file)
            print(f"[{base_name}] {description}]")