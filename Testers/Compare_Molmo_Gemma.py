from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import glob
import os
import time
import ast
import json
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import re

from Prompt.create_classifier_prompt import get_all_bb


def load_molmo(model_id):
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
        use_fast=False
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto"
    )
    return processor, model


def create_pointing_prompt(target_image_path):
    image = Image.open(target_image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                #dict(type="text", text=f"point to all the weapon systems"),
                dict(type="text", text=f"Get all the bounding boxes of the weapon systems"),
                dict(type="image", image=image),
            ],
        }
    ]
    return messages


def run_molmo_prediction(processor, model, prompt):
    inputs = processor.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_tokens      = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text        = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text



def send_to_vllm(client, prompt_func, l_prompt_parms, max_soft_tokens):

    try:
        json_schema, messages = prompt_func(*l_prompt_parms)
        response = client.chat.completions.create(
            model = "google/gemma-4-31B-it",
            messages=messages,
            extra_body={
                "guided_json": json_schema,
                "mm_processor_kwargs": {
                    "max_soft_tokens": max_soft_tokens # 280, 560, 1120
                }
            }
        )
        res_text = response.choices[0].message.content
    except Exception as e:
        print(f"Got an error: {e} FILE: {l_prompt_parms[0]}")
        res_text = "Error"
    return res_text


def run_molmo(file_name, processor, model):

    prompt = create_pointing_prompt(file_name)
    start_time = time.time()
    molmo_res = run_molmo_prediction(processor, model, prompt)
    end_time = time.time()
    return (end_time-start_time), molmo_res


def run_gemma4(file_name):
    MAX_SOFT_TOKENS = 1120  # 280, 560, 1120
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")
    messages = get_all_bb(file_name)
    start_time = time.time()
    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        extra_body={
            "mm_processor_kwargs": {
                "max_soft_tokens": 560#1120
            }
        }
    )
    end_time = time.time()
    res_text = response.choices[0].message.content
    return (end_time-start_time), res_text

def draw_molmo_box(image_path, molmo_res):
    # Load the image
    img           = Image.open(image_path)
    draw          = ImageDraw.Draw(img)
    width, height = img.size
    numbers = [int(n) for n in re.findall(r'\d+', molmo_res)]
    # Filter out the single-digit indices if necessary
    actual_coords = [n for n in numbers if n > 10]
    # 3. Define circle radius
    r = 10

    # 4. Iterate through pairs and draw
    for i in range(0, len(actual_coords), 2):
        # Denormalize coordinates to pixel values
        x = actual_coords[i] * width / 1000
        y = actual_coords[i + 1] * height / 1000

        # Define the bounding box for the ellipse (circle)
        shape = [x - r, y - r, x + r, y + r]
        draw.ellipse(shape, fill="red", outline="red")

    img.show()




if __name__ == "__main__":

    # molmo_res = '<points coords="1 1 460 588 2 504 740">bounding boxes of the weapon systems</points>'
    # file = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_3/frame_294_00_06_912.jpg'
    #
    # molmo_res = '<points coords="1 1 424 595 2 476 870">bounding boxes of the weapon systems</points>'
    # file = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_3/frame_273_00_06_418.jpg'
    # draw_molmo_box(file, molmo_res)
    # exit(0)

    molmo_processor, molmo_model = load_molmo("allenai/Molmo2-4B")
    l_files = glob.glob('/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_3/*.jpg')
    for i in range(3):
        for file in l_files:
            base_name  = os.path.basename(file)
            if base_name != 'frame_273_00_06_418.jpg':
                continue
            elpapse_time, molmo_res = run_molmo(file, molmo_processor, molmo_model )
            print(f"[Molmo] [{elpapse_time:.2f}] [{base_name}] Prediction: {molmo_res}")
            elpapse_time, gemma_res = run_gemma4(file)
            print(f"[Gemma4] [{elpapse_time:.2f}] [{base_name}] Prediction: {gemma_res}")


# Molmo] [1.62] [frame_378_00_08_887.jpg] Prediction: <points coords="1 1 609 959 2 699 959">bounding boxes of the weapon systems</points>
# [Gemma4] [1.27] [frame_378_00_08_887.jpg] Prediction: I did not find any bounding box detections for vehicle weapons.

# molmo
# [1.60] [frame_105_00_02_468.jpg] Prediction: <points coords="1 1 538 652 2 542 279">bounding boxes of the weapon systems</points>
# [1.64] [frame_168_00_03_949.jpg] Prediction: <points coords="1 1 535 652 2 540 669">bounding boxes of the weapon systems</points>
# [1.52] [frame_63_00_01_481.jpg] Prediction: <points coords="1 1 538 659 2 542 279">bounding boxes of the weapon systems</points>
# [1.64] [frame_210_00_04_937.jpg] Prediction: <points coords="1 1 549 382 2 557 414">bounding boxes of the weapon systems</points>

#
# [Molmo] [2.03] [frame_273_00_06_418.jpg] Prediction: <points coords="1 1 424 595 2 476 870">bounding boxes of the weapon systems</points>
# [Gemma4] [10.19] [frame_273_00_06_418.jpg] Prediction: ```json
# [
#   {"box_2d": [382, 466, 422, 483], "label": "vehicle weapon"},
#   {"box_2d": [454, 491, 494, 523], "label": "vehicle weapon"},
#   {"box_2d": [575, 409, 628, 434], "label": "vehicle weapon"},
#   {"box_2d": [671, 506, 711, 524], "label": "vehicle weapon"},
#   {"box_2d": [722, 503, 763, 521], "label": "vehicle weapon"},
#   {"box_2d": [842, 456, 895, 498], "label": "vehicle weapon"}
# ]