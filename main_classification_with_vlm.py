from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import base64
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageDraw
import re
import cv2
import time

from app_config.settings import TEST_CROP_FILES_PATH, FONT_FILE, TEST_FULL_MODE_FILES_PATH, TRAIN_CROP_FILES, \
    TRAIN_FULL_MODE_FILES_PATH


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_description_prompt(target_img):

    # vllm
    base64_img = encode_image(target_img)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image"},
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"},
                },
            ],
        }
    ]
    return messages


def img_to_content(path):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(path)}"
        },
    }

def get_classification_prompt(target_img_path):

    # viewed from a distance
    # a military vehicle in a close-up view,

    sa_22_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/SA-22_CROP_1.jpg"
    sa_22_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/SA-22_CROP_2.jpg"

    scud_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/SCUD_CROP_1.jpg"
    scud_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/SCUD_CROP_2.jpg"
    scud_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/SCUD_CROP_3.jpg"

    t90_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/T-90_CROP_1.jpg"
    t90_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/T-90_CROP_2.jpg"
    t90_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/T-90_CROP_3.jpg"

    sa_22_txt_1 = "The image shows a military weapon system in a close-up view, mounted on a wheeled platform resembling a truck. The structure includes a vehicle-like chassis with multiple wheels for mobility. On both sides of the vehicle, there are mounted artillery cannons, suggesting a mobile firepower system designed for combat or defense purposes. Additionally, a large front window is visible at the front of the vehicle, indicating a driver or operator cabin within the system."
    sa_22_txt_2 = "The image shows a military weapon system viewed from a distance, mounted on a wheeled platform resembling a truck. The structure includes a vehicle-like chassis with multiple wheels for mobility. On both sides of the vehicle, there are mounted artillery cannons, suggesting a mobile firepower system designed for combat or defense purposes. Additionally, a large front window is visible at the front of the vehicle, indicating a driver or operator cabin within the system."

    scud_text_1 = "The image shows a military weapon system in a close-up view, mounted on a truck with wheels for mobility. The truck is used as a ballistic missile launcher and carries a single large, long missile mounted on its platform."
    scud_text_3 = "The image shows a military weapon system viewed from a distance, mounted on a truck with wheels for mobility. The truck is used as a ballistic missile launcher and carries a single large, long missile mounted on its platform."

    t_90_text_1 = "The image shows a tank in a close-up view, with a long, protruding barrel mounted on the turret. In the image there are no wheels, instead it has tracks that allow it to move on terrain. It has a cannon that extends outward beyond the front of the vehicle"
    t_90_text_3 = "The image shows a tank viewed from a distance, with a long, protruding barrel mounted on the turret. In the image there are no wheels, instead it has tracks that allow it to move on terrain. It has a cannon that extends outward beyond the front of the vehicle"

    # VLLM:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify the following images into: Class_1, Class_2, or Class_3."},
                {"type": "text", "text": "Here are examples:"},

                # Shot 1
                img_to_content(sa_22_path_1),
                {"type": "text", "text": sa_22_txt_1+ "\nAnswer: Class_1"},

                img_to_content(sa_22_path_2),
                {"type": "text", "text": sa_22_txt_2 + "\nAnswer: Class_1"},

                # Shot 2
                img_to_content(scud_path_1),
                {"type": "text", "text": scud_text_1+ "\nAnswer: Class_2"},

                img_to_content(scud_path_3),
                {"type": "text", "text": scud_text_3 + "\nAnswer: Class_2"},

                # Shot 3
                img_to_content(t90_path_1),
                {"type": "text", "text": t_90_text_1+ "\nAnswer: Class_3"},

                img_to_content(t90_path_3),
                {"type": "text", "text": t_90_text_3 + "\nAnswer: Class_3"},

                # Query image
                img_to_content(target_img_path),
                {"type": "text",
                 "text": "Based on the examples above, which class does this image belong to? Answer only: Class_1, Class_2, or Class_3."}
            ]
        }
    ]

    return messages


def load_model(model_id):

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    return processor, model


def run_model(processor, model, prompt_func, image_path):
    messages = prompt_func(image_path)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text



def plot_img_with_run_classification(image_path, classifcation):

    img        = Image.open(image_path).convert("RGB")
    draw       = ImageDraw.Draw(img)

    if os.path.exists(FONT_FILE):
        font = ImageFont.truetype(FONT_FILE, size=24)
    else:
        font = ImageFont.load_default()

    draw.text(( 15, 15), classifcation, fill="red", font= font)
    img.show()


def send_to_vllm(client, prompt_func, image_path):
    messages = prompt_func(image_path)
    response = client.chat.completions.create(
        model="/model_path",
        messages=messages
    )
    res_text = response.choices[0].message.content
    return res_text


def run_train_classifcation(client):

    d_convert = {"Class_1": "SA-22",
                 "Class_2": "SCUD",
                 "Class_3": "T-90"}

    l_jpg_file   = []
    l_gt         = []
    l_prediction = []

    df_train_crop = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/embeddings_train_crop.csv')
    df_train_crop = df_train_crop.sample(frac=0.15)
    for i in tqdm(range(len(df_train_crop))):
        jpg_file = df_train_crop.jpg_file.values[i]
        gt       = df_train_crop['gt'].values[i]

        full_crop_file_path = f"{TRAIN_CROP_FILES}{jpg_file}"
        #full_scale_file_path = f"{TRAIN_FULL_MODE_FILES_PATH}{jpg_file}"

        classification = send_to_vllm(client, get_classification_prompt, full_crop_file_path)
        classification = d_convert[classification]

        l_jpg_file .append(jpg_file)
        l_gt .append(gt)
        l_prediction.append(classification)

    df_res = pd.DataFrame({"jpg_file": l_jpg_file, "gt": l_gt, "prediction": l_prediction})
    df_res.to_csv('/home/amitli/repo/dor6_vision/results/train_crop_vlm_classification.csv', index=False)

    print_cm(df_res)

def print_cm(df):
    classes = sorted(df['gt'].unique())
    # compute confusion matrix (counts)
    cm = confusion_matrix(df['gt'], df['prediction'], labels=classes)

    # convert to DataFrame for readability
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    # print("Confusion Matrix (Counts):")
    # print(cm_df)

    cm_percent = cm_df.div(cm_df.sum(axis=1), axis=0) * 100

    print("\nConfusion Matrix (Percentages):")
    print(cm_percent.round(2))


if __name__ == "__main__":

    # T90 as SCUD # '11-18-04_484400_892.jpg', '11-18-04_484400_727.jpg', '11-18-04_484400_435.jpg'
    # SCUD as T-90 # '11-21-10_1324400_1299.jpg', '11-21-10_1324400_1117.jpg', '11-21-10_1324400_1617.jpg'
    # SA-22 as SCUD # 11-20-27_844400_280.jpg, 11-17-44_444400_1275.jpg, 11-17-44_444400_819.jpg
    # SA-22 as T-90 # 11-17-44_444400_1407.jpg, 11-21-02_1244400_1314.jpg, 11-17-44_444400_1626.jpg
    # df = pd.read_csv('/home/amitli/repo/dor6_vision/results/train_crop_vlm_classification.csv')
    # df_sa22_as_scud = df[df['gt'] == 'SA-22']
    # df_sa22_as_scud = df_sa22_as_scud[df_sa22_as_scud.prediction == 'T-90']
    # print(df_sa22_as_scud.jpg_file.values)
    # exit(0)

    # client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")
    # full_crop_file= f"{TRAIN_CROP_FILES}11-17-44_444400_1275.jpg"
    # description = send_to_vllm(client, get_image_description_prompt, full_crop_file)
    # print(description)
    # classification = send_to_vllm(client, get_classification_prompt, full_crop_file)
    # print(classification)
    # exit(0)


    RUN_TRAIN = True

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    if RUN_TRAIN:
        run_train_classifcation(client)
        exit(0)

    df_test_crop  = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/test_set_point.csv')

    #for i in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    for i in [510, 530, 590, 620, 670, 705, 750]:
        #i = 100
        file = df_test_crop['jpg_file'].values[i]
        full_crop_file = f"{TEST_CROP_FILES_PATH}{file}"
        full_size_file = f"{TEST_FULL_MODE_FILES_PATH}{file}"

        d_convert = {"Class_1": "SA-22",
                     "Class_2": "SCUD",
                     "Class_3": "T-90"}

        # description = send_to_vllm(client, get_image_description_prompt, full_crop_file)
        # print(description)
        start_time = time.time()
        classification = send_to_vllm(client, get_classification_prompt, full_crop_file)
        end_time = time.time()
        print(f"[{file}] time = {(end_time - start_time):.2f}s")
        plot_img_with_run_classification(full_size_file, d_convert[classification])
        #print(f"file = {file}")


