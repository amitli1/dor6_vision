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

def get_image_description_prompt(target_img, description_prompt):

    # vllm
    base64_img = encode_image(target_img)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": description_prompt},
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

def get_classification_prompt(target_img_path, extrernal_prompt=None):

    # I want to use gemma3-4B for classifcation between objects (weapon systems) in the image .
    # (The background may change, the weapon system in the image is matter)
    # I want to add few shots example (with those 3 images) (each image contains one weapon systems)
    # please write me description for each image, that I will copy it to the prompt of few shots examples, so the VLM will understand how to classify

    #sa_22_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/11-21-02_1244400_1020.jpg"
    sa_22_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/11-20-27_844400_795.jpg"
    sa_22_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/11-17-44_444400_23.jpg"

    #sa_22_txt_1 = "An overhead view shows a large wheeled military vehicle with a long rectangular chassis and multiple axles. The vehicle carries a rear‑mounted box‑shaped module that occupies most of the vehicle length. The top profile is flat and angular, with no visible gun barrel or turret. The overall silhouette is elongated and truck‑like, indicating a vehicle designed to carry a launcher or payload rather than direct‑fire weapons"
    sa_22_txt_2 = "The image shows a tracked with long, rectangular body with a low profile."
    sa_22_txt_3 = "The image shows a multi-wheeled truck. The front section is a cab with a flat windshield and a narrow profile, with side mirrors. It appears to have six wheels arranged in pairs along the chassis. The rear section has a raised, rectangular launch system holding multiple long, missile-like objects arranged in rows. There also appear to be exhaust pipes on the sides"

    scud_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/11-21-10_1324400_673.jpg"
    scud_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/11-17-54_524400_136.jpg"
    scud_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/11-20-34_884400_1224.jpg"

    scud_txt_1 = "The image shows a long, rectangular vehicle with a multi-wheeled chassis. It has a cab at the front and a long, enclosed cargo or equipment section behind it."
    scud_txt_2 = "The image shows a long, rectangular vehicle with a cab at the front. A long cylindrical object (likely a missile) is mounted on top, extending significantly beyond the cab. The vehicle has multiple wheels—at least six are visible—arranged in pairs along its length"
    scud_txt_3 = "The image shows a long, rectangular vehicle with a cylindrical object (likely a missile) mounted on top."

    t_90_path_1 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/11-20-40_924400_731.jpg"
    #t_90_path_2 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/11-18-04_484400_633.jpg"
    t_90_path_3 = "/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/11-21-17_1284400_311.jpg"

    t_90_txt_1 = "The image features a modern main battle tank with a distinct low-profile, rounded (hemispherical) turret. Centered in the turret is a long smoothbore main gun, often featuring a thermal sleeve or fume extractor. The hull is elongated and sits low to the ground, protected by thick frontal glacis armor, with heavy side skirts covering the upper portion of the tracks."
    #t_90_txt_2 = "A main battle tank featuring a large, boxy turret with sharp, angled surfaces designed for kinetic energy deflection. It is armed with a large-caliber smoothbore gun. The exterior is notable for modular or composite armor plates bolted onto the turret faces and hull, creating a multi-layered, geometric appearance compared to cast-steel designs."
    t_90_txt_3 = "A high-angle view of a main battle tank characterized by its continuous caterpillar tracks and a heavy armored hull. The most prominent feature is the centrally mounted, 360-degree rotating turret which houses a long-barrelled primary cannon. The silhouette is defined by the mechanical complexity of the drive sprockets and the low-profile chassis."


    # VLLM:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You receive an image from a simulation and you must classify the military vehicle in the image, based only on the vehicle structure and mounted weapon system. Ignore background, terrain, and camera angle."},
                {"type": "text", "text": "Identify the object in the frame. If the object is small or distant, consider its overall shape, color patterns."},
                {"type": "text", "text": "Examples:"},

                # Shot 1
                #img_to_content(sa_22_path_1),
                #{"type": "text", "text": sa_22_txt_1+ "\nAnswer: class_1"},

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

                # img_to_content(t_90_path_2),
                # {"type": "text", "text": t_90_txt_2+ "\nAnswer: class_3"},

                img_to_content(t_90_path_3),
                {"type": "text", "text": t_90_txt_3+ "\nAnswer: class_3"},


                # Query image
                img_to_content(target_img_path),
                # {"type": "text",
                #  "text": "Analyze the provided image. First, describe the primary object's shape and visible features. Second, based on those features, classify the object into one of the following categories"},
                {"type": "text",
#                 "text": "Based on the examples above, which class does this image belong to? Answer only: 'class_1', 'class_2', 'class_3' or 'Nothing'."}
                   "text": "Based on the examples above, which class does this image belong to? If the image does not fit any of the three, answer 'none'. Answer only: 'class_1', 'class_2', 'class_3', or 'none'."}
            ]
        }
    ]

    return messages




def plot_img_with_run_classification(image_path, classifcation):

    img        = Image.open(image_path).convert("RGB")
    draw       = ImageDraw.Draw(img)

    if os.path.exists(FONT_FILE):
        font = ImageFont.truetype(FONT_FILE, size=24)
    else:
        font = ImageFont.load_default()

    draw.text(( 15, 15), classifcation, fill="red", font= font)
    img.show()


def send_to_vllm(client, prompt_func, image_path, external_prompt=None):
    messages = prompt_func(image_path, external_prompt)
    response = client.chat.completions.create(
        #model="/model_path",
        model = "google/gemma-4-31B-it",
        messages=messages,
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
    df_train_crop = df_train_crop.sample(frac=0.50)
    for i in tqdm(range(len(df_train_crop))):
        jpg_file = df_train_crop.jpg_file.values[i]
        gt       = df_train_crop['gt'].values[i]

        full_crop_file_path = f"{TRAIN_CROP_FILES}{jpg_file}"
        #full_scale_file_path = f"{TRAIN_FULL_MODE_FILES_PATH}{jpg_file}"

        classification = send_to_vllm(client, get_classification_prompt, full_crop_file_path)
        classification = classification.strip()
        if classification.find('Answer:') != -1:
            classification = classification[7:].strip()
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


def eda_few_shots(client):
    import glob
    l_sa_22 = glob.glob('/home/amitli/repo/dor6_vision/Dataset/few_shots/SA-22/*.jpg')
    l_scud = glob.glob('/home/amitli/repo/dor6_vision/Dataset/few_shots/SCUD/*.jpg')
    l_t_90 = glob.glob('/home/amitli/repo/dor6_vision/Dataset/few_shots/T-90/*.jpg')

    l_all = l_sa_22 + l_scud + l_t_90
    l_gt  = ['SA-22'] * 3 + ['SCUD'] * 3 + ['T-90'] * 3

    for i in range(len(l_all)):
        print("\n----------------------------------------------------------------------\n")
        file        = l_all[i]
        gt          = l_gt[i]
        desc_prompt = f"You are getting a picture from a simulation that contains a {gt} military vehicle. Describe ONLY the visual features of the military vehicle in the picture (Only the visual features you see in this image)."
        description = send_to_vllm(client, get_image_description_prompt, file, desc_prompt)
        print(f"[{gt}] {os.path.basename(file)} = {description}")




if __name__ == "__main__":

    RUN_TRAIN = True

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    # eda_few_shots(client)
    # exit(0)


    if RUN_TRAIN:
        run_train_classifcation(client)
        exit(0)


    RUN_ON_TEST_SET = False
    if RUN_ON_TEST_SET:
        df_test_crop  = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/test_set_point.csv')
        for i in range(len(df_test_crop)):
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


 # Confusion Matrix (Percentages):
    #        SA-22   SCUD   T-90
    # SA-22  34.02   9.31  56.66
    # SCUD    1.51  63.43  35.06
    # T-90    0.00   0.51  99.49

    #  gemma 4 - no nOne ((15%))
    # Confusion Matrix (Percentages):
    #        SA-22   SCUD   T-90
    # SA-22  93.35   3.58   3.07
    # SCUD    9.75  88.00   2.25
    # T-90    4.99   0.00  95.01

    #  gemma 4 - with nOne (15%)
    # /home/amitli/repo/dor6_vision/.venv/bin/python /home/amitli/repo/dor6_vision/main_classification_with_vlm.py
    #   8%|▊         | 198/2364 [02:09<20:40,  1.75it/s]11-17-44_444400_696.jpg = none
    #  24%|██▎       | 561/2364 [06:07<18:10,  1.65it/s]11-21-02_1244400_967.jpg = none
    #  29%|██▉       | 693/2364 [07:36<22:58,  1.21it/s]11-18-04_484400_193.jpg = Please provide the image you would like me to classify.
    #  41%|████      | 959/2364 [10:35<14:56,  1.57it/s]11-20-27_844400_323.jpg = none
    #  48%|████▊     | 1126/2364 [12:30<13:50,  1.49it/s]11-21-02_1244400_1327.jpg = none
    #  49%|████▊     | 1150/2364 [12:46<13:24,  1.51it/s]11-21-02_1244400_1341.jpg = none
    #  59%|█████▉    | 1394/2364 [15:02<10:38,  1.52it/s]11-21-17_1284400_92.jpg = Please provide the image you would like me to classify.
    #  88%|████████▊ | 2073/2364 [21:02<02:09,  2.24it/s]11-21-02_1244400_1527.jpg = none
    # 100%|██████████| 2364/2364 [23:23<00:00,  1.68it/s]
    #
    # Confusion Matrix (Percentages):
    #        SA-22   SCUD   T-90
    # SA-22  90.86   4.82   4.31
    # SCUD    6.35  92.45   1.20
    # T-90    3.27   0.14  96.59

    #gemma 4 - with nOne (15%)
    # /home/amitli/repo/dor6_vision/.venv/bin/python /home/amitli/repo/dor6_vision/main_classification_with_vlm.py
    #  10%|▉         | 229/2364 [02:01<19:36,  1.81it/s]11-17-44_444400_291.jpg = none
    #  31%|███       | 733/2364 [06:37<16:23,  1.66it/s]11-21-02_1244400_1156.jpg = none
    #  35%|███▍      | 821/2364 [07:25<13:42,  1.88it/s]11-20-40_924400_960.jpg = none
    #  57%|█████▋    | 1359/2364 [12:13<08:29,  1.97it/s]11-21-02_1244400_18.jpg = none
    #  83%|████████▎ | 1955/2364 [17:29<03:19,  2.05it/s]11-21-02_1244400_967.jpg = none
    # 100%|██████████| 2364/2364 [21:04<00:00,  1.87it/s]
    #
    # Confusion Matrix (Percentages):
    #        SA-22   SCUD   T-90
    # SA-22  91.52   3.37   5.11
    # SCUD    5.97  92.04   1.99
    # T-90    2.92   0.00  97.08