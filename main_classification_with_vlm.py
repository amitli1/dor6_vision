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

    # I want to use gemma3-4B for classifcation between objects (weapon systems) in the image .
    # (The background may change, the weapon system in the image is matter)
    # I want to add few shots example (with those 3 images) (each image contains one weapon systems)
    # please write me description for each image, that I will copy it to the prompt of few shots examples, so the VLM will understand how to classify

    sa_22_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/11-21-02_1244400_1020.jpg"
    sa_22_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/11-20-27_844400_795.jpg"
    sa_22_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/11-17-44_444400_588.jpg"

    scud_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/11-21-10_1324400_673.jpg"
    scud_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/11-17-54_524400_136.jpg"
    scud_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/11-20-34_884400_1224.jpg"

    t_90_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/11-20-40_924400_731.jpg"
    t_90_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/11-18-04_484400_633.jpg"
    t_90_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/11-21-17_1284400_1530.jpg"

    sa_22_txt_1 = "An overhead view shows a large wheeled military vehicle with a long rectangular chassis and multiple axles. The vehicle carries a rear‑mounted box‑shaped module that occupies most of the vehicle length. The top profile is flat and angular, with no visible gun barrel or turret. The overall silhouette is elongated and truck‑like, indicating a vehicle designed to carry a launcher or payload rather than direct‑fire weapons"
    sa_22_txt_2 = "An overhead image shows a large wheeled military vehicle with a long, rectangular truck‑like chassis and multiple axles. A box‑shaped rear superstructure occupies much of the vehicle length. The top profile is flat and angular, with no visible turret or gun barrel. The vehicle’s silhouette is elongated and modular, dominated by a rear-mounted payload rather than a compact fighting compartment."
    sa_22_txt_3 = "An overhead image shows a wheeled military vehicle with an elongated rectangular chassis and multiple axles. The vehicle has a large, box‑shaped rear superstructure occupying much of the vehicle length. The top surface is flat and angular, with no visible turret, cannon, or forward‑pointing gun barrels. The overall silhouette is long and modular, dominated by a rear-mounted payload rather than a compact fighting compartment."

    scud_txt_1 = "An aerial image depicts a multi‑axle wheeled vehicle with a cylindrical launcher tube mounted along the centerline of the chassis. The launcher is long, rounded, and raised above the vehicle body, extending nearly the full length of the platform. The vehicle has a distinct separation between the driving cab and the launcher system. No gun barrels or rotating turret are visible"
    scud_txt_2 = "An aerial view depicts a multi‑axle wheeled vehicle carrying a single long cylindrical launcher tube mounted along the vehicle’s centerline. The tube is rounded, smooth, and elevated above the chassis, extending nearly the full length of the vehicle. The launcher is visually distinct from the cab area. No turret or direct‑fire gun barrels are visible"
    scud_txt_3 = "An aerial view depicts a multi‑axle wheeled vehicle carrying a single long cylindrical launcher tube mounted along the vehicle’s centerline. The tube is rounded, uniform in diameter, and elevated above the chassis, extending nearly the full length of the vehicle. The launcher structure is visually distinct from the cab area. There are no rotating turrets or direct‑fire gun barrels visible."

    t_90_txt_1 = "Top‑down aerial image of a tracked or heavy wheeled armored vehicle traveling on a paved road. The vehicle features a central rotating turret mounted on top of the hull. A long gun barrel extends forward from the turret, clearly visible and projecting beyond the front of the vehicle. The hull is compact and rectangular, significantly shorter than launcher trucks. The defining feature is the turret‑mounted cannon rather than a rear launcher system."
    t_90_txt_2 = "Aerial image of an armored vehicle positioned near a shoreline. The vehicle has a solid armored hull and a turret mounted centrally on top. A gun barrel extends outward from the turret, forming a distinct protruding weapon. The vehicle is compact compared to long launcher trucks and lacks cylindrical missile tubes or large rear containers. The turret and barrel are the primary identifying elements."
    t_90_txt_3 = "Top‑down aerial view of an armored combat vehicle on flat, open terrain. The vehicle has a low‑profile hull with a centrally mounted turret. One or more gun barrels are visible extending from the turret. The overall shape is compact and dense, with armor plates and no elongated launcher structures. The presence of a turret and direct‑fire gun differentiates this vehicle from launcher or missile carrier systems."


    # VLLM:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify the weapon system based only on the vehicle structure and mounted weapon system. Ignore background, terrain, and camera angle."},
                {"type": "text", "text": "Examples:"},

                # Shot 1
                img_to_content(sa_22_path_1),
                {"type": "text", "text": sa_22_txt_1+ "\nAnswer: class_1"},

                img_to_content(sa_22_path_2),
                {"type": "text", "text": sa_22_txt_2 + "\nAnswer: class_1"},

                img_to_content(sa_22_path_3),
                {"type": "text", "text": sa_22_txt_3 + "\nAnswer: class_1"},

                # Shot 2
                img_to_content(scud_path_1),
                {"type": "text", "text": scud_txt_1+ "\nAnswer: class_2"},

                img_to_content(scud_path_2),
                {"type": "text", "text": scud_txt_2 + "\nAnswer: class_2"},

                img_to_content(scud_path_3),
                {"type": "text", "text": scud_txt_3 + "\nAnswer: class_2"},

                # Shot 3
                img_to_content(t_90_path_1),
                {"type": "text", "text": t_90_txt_1+ "\nAnswer: class_3"},

                img_to_content(t_90_path_2),
                {"type": "text", "text": t_90_txt_2+ "\nAnswer: class_3"},

                img_to_content(t_90_path_3),
                {"type": "text", "text": t_90_txt_3+ "\nAnswer: class_3"},


                # Query image
                img_to_content(target_img_path),
                {"type": "text",
                 "text": "Based on the examples above, which class does this image belong to? Answer only: 'class_1', 'class_2', or 'class_3'."}
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

    d_convert = {"class_1": "SA-22",
                 "class_2": "SCUD",
                 "class_3": "T-90"}

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
        if classification not in d_convert.keys():
            print(f"{jpg_file} = {classification}")
        else:
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


