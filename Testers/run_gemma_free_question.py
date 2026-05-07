import cv2
from openai                          import OpenAI
import base64
import json

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def add_text_line(content, text):
    content.append({"type": "text", "text": f"{text}"})

def add_image_line(content, image_path):
    content.append({"type": "image_url", "image_url": {"url":f"data:image/jpeg;base64,{encode_image(image_path)}"}})


def build_prompt(target_img, question):
    base_prompt = question
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


if __name__ == "__main__":

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")
    full_file_path = '/home/amitli/repo/dor6_vision/Testers/tmp_files/SA-6.jpg'
    full_file_path = '/home/amitli/repo/dor6_vision/Testers/tmp_files/TRUCK.jpg'
    full_file_path = '/home/amitli/repo/dor6_vision/Testers/tmp_files/NOT_SCUD_CROP_1.jpg'

    messages = build_prompt(full_file_path, "Is this SCUD ?")
    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        extra_body={
            "mm_processor_kwargs": {
                "max_soft_tokens": 140
            }
        }
    )
    res_text = response.choices[0].message.content
    print(res_text)