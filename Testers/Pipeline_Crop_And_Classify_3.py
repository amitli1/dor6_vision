from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from app_config.settings import FONT_FILE, TRAIN_FULL_MODE_FILES_PATH
from main_classification_with_vlm import print_cm
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import numpy as np

import re
import ast
import time
import json
import os
import glob
import base64

TMP_FILES_FOLDER = '/home/amitli/repo/dor6_vision/Testers/tmp_files'
BB_TMP_FILE = "/home/amitli/repo/dor6_vision/Testers/tmp_files/bb.json"
#MODEL = "gemma-4-31b-it" #MODEL = google/gemma-4-31B-it"
MODEL = "allenai/Molmo2-4B"
BASE_URL = "http://localhost:9100/v1"
#BASE_URL = "http://localhost:9000/v1"


def draw_box(image_path, l_cords, output_jpg_file=None, l_prediction=None, show_img=False):
    # Load the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    font = ImageFont.truetype(FONT_FILE, size=32)

    # pred_str = ""
    # for i in range(len(l_prediction)):
    #     pred_str += l_prediction[i] + "\n"
    # draw.text((1, 1), pred_str, fill="red", font=font)

    if type(l_cords) == str:
        l_cords = ast.literal_eval(l_cords)

    for i, coords in enumerate(l_cords):
        ymin, xmin, ymax, xmax = coords

        # 2. Convert from normalized (0-1000) to actual pixel values
        left = xmin * width / 1000
        top = ymin * height / 1000
        right = xmax * width / 1000
        bottom = ymax * height / 1000

        # 3. Draw the rectangle
        # PIL expects [xmin, ymin, xmax, ymax]
        draw.rectangle([left, top, right, bottom], outline="red", width=1)
        if l_prediction is not None:
            draw.text((left + 50, top), l_prediction[i].replace('none', 'Other'), fill="red", font=font)
        # draw.text((left+50, top), f"{i+1}", fill="red", font=font)

    if output_jpg_file:
        img.save(output_jpg_file)
    if show_img:
        img.show()


def get_all_points(target_img):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Locate all the military vehicles in this image. Return only their locations"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(target_img)}"}}
            ],
        }
    ]
    return messages


def extract_molmo_coords(text):

    # Regex searches for the content inside 'coords="..."'
    match = re.search(r'coords="([\d\s]+)"', text)

    if match:
        # Get the string of numbers
        coord_string = match.group(1)
        # Split by whitespace and convert to integers
        coords = [int(c) for c in coord_string.split()]
        return coords

    return []


def convert_point_to_boxes(coords, offset=100):
    boxes = []

    # Molmo points are structured as [ID, Y, X]
    # We iterate in steps of 3
    for i in range(2, len(coords), 3):

        y_center = coords[i]
        x_center = coords[i + 1]

        # Calculate boundaries with a 0-1000 clamp
        ymin = max(0, y_center - offset)
        xmin = max(0, x_center - offset)
        ymax = min(1000, y_center + offset)
        xmax = min(1000, x_center + offset)

        # Append as a dictionary (JSON object)
        boxes.append({
            "box_2d": [ymin, xmin, ymax, xmax]
        })

    return boxes


def get_list_of_points(client, full_image_path):
    messages = get_all_points(full_image_path)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        top_p=1,
        seed=42,
        # extra_body={
        #     "guided_json": schema,
        #
        # }
    )
    res_text = response.choices[0].message.content
    res_text = extract_molmo_coords(res_text)
    #res_json = convert_point_to_boxes(res_text)

    return res_text


def create_crop_files(image_path, model_bb_json_res, min_crop_size):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    # draw          = ImageDraw.Draw(image)
    l_crop_ratio = []

    for i, bb in enumerate(model_bb_json_res):
        ymin, xmin, ymax, xmax = bb['box_2d']
        left = (xmin / 1000) * width
        top = (ymin / 1000) * height
        right = (xmax / 1000) * width
        bottom = (ymax / 1000) * height

        # draw.rectangle([left, top, right, bottom], outline="red", width=3)
        # crop_image = image.crop((left, top, right, bottom))
        # crop_image.show()

        crop_width = int(right - left)
        crop_height = int(bottom - top)
        crop_size = (crop_width * crop_height) / (width * height)
        l_crop_ratio.append(crop_size)

        if crop_width < min_crop_size:
            delta = (min_crop_size - crop_width) / 2
            left -= delta
            right += delta

        if crop_height < min_crop_size:
            delta = (min_crop_size - crop_height) / 2
            top -= delta
            bottom += delta

        crop_image = image.crop((left, top, right, bottom))
        # crop_image.show()

        # print(f"[{os.path.basename(image_path)}] Image size: {image.size}, Crop #{i+1} size = ({crop_width},{crop_height}) [{crop_size:.3f}%]")
        crop_image.save(f"{TMP_FILES_FOLDER}/crop_{i + 1}.jpg", "JPEG")

    # image.show()
    return l_crop_ratio


def classify_molmo_objects(client, objects_path, num_of_objects):
    messages = get_classification_prompt(objects_path, num_of_objects)

    schema = {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "visual_evidence": {
                            "type": "string",
                            "description": "List seen features: wheels/tracks, number/placement of missiles."
                        },
                        "classification": {
                            "type": "string",
                            "enum": ["Class_1", "Class_2", "Class_3", "none", "Uncertain"]
                        }
                    },
                    "required": ["visual_evidence", "classification"]
                }
            }
        },
        "required": ["images"]
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        top_p=1,
        seed=42,
        extra_body={
            "guided_json": schema,
        }
    )
    res_text = response.choices[0].message.content
    return res_text


def classify_objects(client, objects_path, num_of_objects):
    messages = get_classification_prompt(objects_path, num_of_objects)

    schema = {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "visual_evidence": {
                            "type": "string",
                            "description": "List seen features: wheels/tracks, number/placement of missiles."
                        },
                        "classification": {
                            "type": "string",
                            "enum": ["Class_1", "Class_2", "Class_3", "none", "Uncertain"]
                        }
                    },
                    "required": ["visual_evidence", "classification"]
                }
            }
        },
        "required": ["images"]
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        top_p=1,
        seed=42,
        extra_body={
            "guided_json": schema,
            "mm_processor_kwargs": {
                "max_soft_tokens": 280  # 70, 140, 280
            }
        }
    )
    res_text = response.choices[0].message.content
    return res_text


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def add_text_line(content, text):
    content.append({"type": "text", "text": f"{text}"})


def add_image_line(content, image_path):
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})


def get_classification_prompt(objects_path, num_of_objects):
    FEW_SHOTS_FOLDER = '/home/amitli/repo/dor6_vision/Testers/few_shots'
    # SA_22 = "SA-22"
    SA_22 = "Class_1"
    # SCUD  = "SCUD"
    SCUD = "Class_2"
    # T_90  = "T-90"
    T_90 = "Class_3"

    sa22_txt_1 = "Multi-axle truck chassis with four visible wheels on one side; rear bed carries a large rectangular module supporting multiple parallel cylindrical launch tubes"
    sa22_txt_2 = "Rectangular vehicle chassis with a large flatbed mounting two parallel, elongated cylindrical launch tubes on the rear section."
    sa22_txt_3 = "Rectangular chassis with a rear-mounted multi-tube rocket launcher assembly and a large circular component situated between the tube banks."

    scud_txt_1 = "Long rectangular chassis with multiple wheels, topped with a long, cylindrical launch tube extending along the length of the vehicle."
    # scud_txt_2 = "Tall, vertical cylindrical launch tubes mounted on a rectangular base chassis."
    scud_txt_3 = "A heavy-duty vehicle with multiple axles with a long, cylindrical tube (missile) mounted on top of the chassis"

    t90_txt_1 = "Tracked chassis with a centrally mounted turret and a long protruding gun barrel."
    t90_txt_2 = "Rectangular hull with tracks, featuring a top-mounted turret and a long projecting barrel."
    t90_txt_3 = "Tracked chassis with a rectangular hull, centrally mounted rotating turret, and a long forward-facing main gun barrel."

    content = []
    add_text_line(content,'You are an expert in identifying military vehicles from simulation images')
    add_text_line(content,f"You receive {num_of_objects} patches of images from a military simulation and you must classify the military vehicle in the image, based only on the vehicle structure and mounted weapon system. Ignore background, terrain, and camera angle.")
    add_text_line(content,"CRITICAL RULE: Accuracy is more important than identification. If structural features are missing or ambiguous, you MUST label as 'Uncertain' or 'none'.")
    add_text_line(content,"Base your decision ONLY on visible structural features (e.g., wheels vs tracks, turret type, missile placement, number of missiles).")
    add_text_line(content, "Do NOT rely on color, background")

    add_text_line(content, "VISUAL DIFFERENTIATORS")

    add_text_line(content, f"{SA_22}")
    add_text_line(content, "1. A truck‑mounted system")
    add_text_line(content, "2. Has a visible dual autocannons")
    add_text_line(content, "3. Has a radar module")
    add_text_line(content, "4. The missiles are mounted on the sides of the turret, not in the center.")
    add_text_line(content, "5. Has cabin at the front and flatbed rear layout")
    add_text_line(content, "6. It must have missiles on it")  # new

    add_text_line(content, f"{SCUD}")
    add_text_line(content, "1. carry one large ballistic missile")
    add_text_line(content, "2. Very long TEL truck")
    add_text_line(content, "3. No radar antennas")
    add_text_line(content, "4. Definitively identify one missile")

    add_text_line(content, f"{T_90}")
    add_text_line(content, "1. TANK")
    add_text_line(content, "2. Large gun turret with a single main cannon")
    add_text_line(content, "3. Prominent rotating turret")
    add_text_line(content, "4. Tracks instead of wheels")

    add_text_line(content, f"Reject {SA_22} if:")
    add_text_line(content, "1. The missiles are NOT mounted on the sides of the turret")
    add_text_line(content, "2. Contains one missile")
    add_text_line(content, "3. Launch tubes protrude out of the front of the truck")
    add_text_line(content, "4. No clear missiles in the image")
    add_text_line(content, "5. Three long missiles on a launcher arm")

    add_text_line(content, f"Reject {T_90} if:")
    add_text_line(content, "1. missile launcher")
    add_text_line(content, "2. There is no turret")

    add_text_line(content, f"Reject {SCUD} if:")
    add_text_line(content, "1. More than one missile")
    add_text_line(content, "2. There is no missile")

    add_text_line(content, "Examples:")

    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SA-22_1.JPG')
    add_text_line(content, f"{sa22_txt_1} Answer: {SA_22}")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SA-22_2.JPG')
    add_text_line(content, f"{sa22_txt_2} Answer: {SA_22}")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SA-22_3.JPG')
    add_text_line(content, f"{sa22_txt_3} Answer: {SA_22}")

    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SCUD_1.JPG')
    add_text_line(content, f"{scud_txt_1} Answer: {SCUD}")
    # add_image_line(content, f'{FEW_SHOTS_FOLDER}/SCUD_2.JPG')
    # add_text_line(content, f"{scud_txt_2} Answer: {SCUD}")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SCUD_3.JPG')
    add_text_line(content, f"{scud_txt_3} Answer: {SCUD}")

    add_image_line(content, f'{FEW_SHOTS_FOLDER}/T-90_1.JPG')
    add_text_line(content, f"{t90_txt_1} Answer: {T_90}")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/T-90_2.JPG')
    add_text_line(content, f"{t90_txt_2} Answer: {T_90}")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/T-90_3.JPG')
    add_text_line(content, f"{t90_txt_3} Answer: {T_90}")

    # --- no guess
    # add_text_line(content, 'If an image is too blurry to identify, label it as "Uncertain"')
    add_text_line(content, "If a clear object from the known classes is visible → output the class")
    add_text_line(content, 'If multiple classes are plausible or visibility is poor → output "Uncertain"')
    add_text_line(content, "Do not guess. If you are not sure with high certainty → output 'Uncertain'")

    add_text_line(content, f"TASK: Classify the following {num_of_objects} images. For each image:")
    add_text_line(content, "1. Describe what you see: 'I see [wheels/tracks], [missiles/gun], [radar/no radar]'.")
    add_text_line(content, "2. State if it meets all criteria or meets a REJECT condition.")
    add_text_line(content, f"3. Provide final classification: ['{SA_22}', '{SCUD}', '{T_90}', 'none', 'Uncertain'].")

    # 5. RESPONSE FORMAT INSTRUCTIONS
    add_text_line(content, "RESPONSE FORMAT:")
    add_text_line(content, "You must return a JSON object containing an array of 'images'.")
    add_text_line(content,"For each image, you MUST first provide 'visual_evidence' describing the wheels/tracks and weapon systems before giving the 'classification'.")

    for i in range(num_of_objects):
        add_text_line(content, f"Image: {i + 1}")
        crop_file_path = f"{objects_path}/crop_{i + 1}.jpg"
        add_image_line(content, crop_file_path)

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


def simulate_vlm_view(image_path, target_res=(224, 224), patch_size=16):
    img = Image.open(image_path)

    # 1. Upsample to the model's expected internal resolution
    # Gemma 4 usually uses bilinear or bicubic interpolation
    img_resized = img.resize(target_res, Image.Resampling.BICUBIC)

    # 2. Visualize the Patch Grid
    # The model "sees" the image as a sequence of these squares
    arr = np.array(img_resized)
    for i in range(0, target_res[0], patch_size):
        arr[i:i + 1, :, :] = [255, 0, 0]  # Red horizontal grid lines
    for j in range(0, target_res[1], patch_size):
        arr[:, j:j + 1, :] = [255, 0, 0]  # Red vertical grid lines

    return Image.fromarray(arr)


def update_prediction(prediction):
    if prediction.lower().find('class') == -1:
        return prediction
    else:
        if prediction.lower() == 'class_1':
            return "SA-22"
        if prediction.lower() == 'class_2':
            return "SCUD"
        if prediction.lower() == 'class_3':
            return "T-90"
        return prediction


def test_on_train():

    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)

    #df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/labels_balanced_test_500.csv')
    df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/shiry_testset_balanced.csv')
    #df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/labels_balanced_test_300.csv')
    df = df.rename(columns={'filename': 'jpg_file', 'label_name': 'gt'})
    df = df[df['gt'] != 'Other']
    #df = df.groupby('gt', group_keys=False).sample(frac=0.2, random_state=42)
    #df.to_csv('/home/amitli/repo/dor6_vision/Dataset/labels_balanced_test_200.csv', index=False)

    l_jpg_file = []
    l_gt = []
    l_prediction = []
    l_description = []
    l_time = []
    l_class_time = []
    l_crop_ratio = []
    l_crop_values = []

    for i in tqdm(range(len(df))):
        jpg_file = df['jpg_file'].values[i]
        gt = df['gt'].values[i]
        start_time = time.time()

        try:
            full_image_path = f"{TRAIN_FULL_MODE_FILES_PATH}{jpg_file}"
            model_json_res = get_list_of_points(client, full_image_path)
            with open(BB_TMP_FILE, "w") as f:
                json.dump(model_json_res, f, indent=4)
            crop_ratio           = create_crop_files(full_image_path, model_json_res, 128)
            start_cls_time       = time.time()
            classifcation_result = classify_objects(client, TMP_FILES_FOLDER, len(model_json_res))
            end_cls_time         = time.time()

            classifcation_result = classifcation_result.replace("```json", "").replace("```", "").strip()
            classifcation_result = json.loads(classifcation_result)
            prediction          = classifcation_result['images'][0]['classification']

            prediction   = update_prediction(prediction)
            description  = classifcation_result['images'][0]['visual_evidence']
        except Exception as e:
            print(f"Error in {jpg_file}")
            prediction = "Error"
            description = "Error"
            crop_ratio = "Error"
            model_json_res = "Error"
            start_cls_time = 0
            end_cls_time = 0

        end_time = time.time()
        l_jpg_file.append(jpg_file)
        l_gt.append(gt)
        l_prediction.append(prediction)
        l_description.append(description)
        l_time.append(end_time - start_time)
        l_crop_ratio.append(crop_ratio)
        l_crop_values.append(model_json_res)
        l_class_time.append(end_cls_time - start_cls_time)

    df_tmp = pd.DataFrame({"jpg_file": l_jpg_file,
                           'gt': l_gt,
                           'prediction': l_prediction,
                           'description': l_description,
                           "total_diff_time": l_time,
                           'crop_ratio': l_crop_ratio,
                           'crop_values': l_crop_values,
                           "class_time": l_class_time})

    df_tmp.to_csv('150_molmo.csv', index=False)
    print_cm(df_tmp)


def run_pipeline(client, full_image_path):
    GET_LIST_OF_BB    = True
    CREATE_CROP_FILES = True
    CLASSIFY          = True

    if GET_LIST_OF_BB:
        start_time = time.time()
        model_json_res = get_list_of_points(client, full_image_path)
        end_time = time.time()
        with open(BB_TMP_FILE, "w") as f:
            json.dump(model_json_res, f, indent=4)
        print(f"[{(end_time - start_time):.2f} sec] Get list of BB")
    if CREATE_CROP_FILES:
        with open(BB_TMP_FILE, "r") as f:
            model_json_res = json.load(f)
        start_time = time.time()
        create_crop_files(full_image_path, model_json_res, 128)
        end_time = time.time()
        print(f"[{(end_time - start_time):.2f} sec] Create crop files")
    if CLASSIFY:
        with open(BB_TMP_FILE, "r") as f:
            model_json_res = json.load(f)
        start_time = time.time()
        classifcation_result   = classify_objects(client, TMP_FILES_FOLDER, len(model_json_res))
        classifcation_result   = classifcation_result.replace("```json", "").replace("```", "").strip()
        l_classifcation_result = json.loads(classifcation_result)

        end_time = time.time()

        l_prediction = []
        l_bb = []
        for i in range(len(model_json_res)):

            classification  = l_classifcation_result['images'][i]['classification']
            classification = update_prediction(classification)
            description    = l_classifcation_result['images'][i]['visual_evidence']


            l_prediction.append(f"[{i + 1}] {classification}")
            l_bb.append(model_json_res[i]['box_2d'])
            print(f'\t[{i + 1}] [{classification}] {description}')

        print(f"[{(end_time - start_time):.2f} sec] Object classification")

    return l_prediction, l_bb


def plot_time_avg(df):
    import matplotlib.pyplot as plt
    avg_time_per_obj = df.groupby('num_objs_in_img')['time_diff'].mean()

    # make sure x is sorted
    avg_time_per_obj = avg_time_per_obj.sort_index()

    # plot
    plt.figure()
    plt.plot(avg_time_per_obj.index, avg_time_per_obj.values, marker='o')
    plt.xlabel('Number of objects in image')
    plt.ylabel('Average time_diff')
    plt.title('Average time_diff vs number of objects')
    plt.grid(True)
    plt.show()


def draw_with_molmo_point(full_image_path, model_point_output):
    img = Image.open(full_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    radius = 10  # Size of the circle

    # Iterate in steps of 3: [ID, Y, X]
    # model_output =  [1, 1, 89, 844, 2, 289, 124, 3, 338, 339]
    for i in range(2, len(model_point_output), 3):
        x_norm = model_point_output[i]
        y_norm = model_point_output[i + 1]

        # Scale coordinates
        pixel_x = (x_norm / 1000) * w
        pixel_y = (y_norm / 1000) * h

        left_up = (pixel_x - radius, pixel_y - radius)
        right_down = (pixel_x + radius, pixel_y + radius)

        draw.ellipse([left_up, right_down], fill="red", outline="white")

    img.show()

def create_molmo_crop_files(full_image_path, model_point_output, radius):
    img = Image.open(full_image_path).convert("RGB")
    w, h = img.size
    crop_index = 1
    for i in range(2, len(model_point_output), 3):
        x_norm = model_point_output[i]
        y_norm = model_point_output[i + 1]

        # Scale coordinates
        pixel_x = (x_norm / 1000) * w
        pixel_y = (y_norm / 1000) * h

        left = pixel_x -radius
        top = pixel_y - radius
        right = pixel_x + radius
        bottom = pixel_y + radius

        crop_image = img.crop((left, top, right, bottom))
        crop_image.save(f"{TMP_FILES_FOLDER}/crop_{crop_index}.jpg", "JPEG")
        crop_index = crop_index + 1


def draw_with_molmo_bb(full_image_path, model_point_output):
    img = Image.open(full_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    radius = 100  # Size of the circle

    # Iterate in steps of 3: [ID, Y, X]
    # model_output =  [1, 1, 89, 844, 2, 289, 124, 3, 338, 339]
    for i in range(2, len(model_point_output), 3):
        x_norm = model_point_output[i]
        y_norm = model_point_output[i + 1]

        # Scale coordinates
        pixel_x = (x_norm / 1000) * w
        pixel_y = (y_norm / 1000) * h

        left = pixel_x -radius
        top = pixel_y - radius
        right = pixel_x + radius
        bottom = pixel_y + radius

        # 3. Draw the rectangle
        # PIL expects [xmin, ymin, xmax, ymax]
        draw.rectangle([left, top, right, bottom], outline="red", width=1)


    img.show()


def test_molmo_on_train():
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)


    # df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/labels_balanced_test_500.csv')
    df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/shiry_testset_balanced.csv')
    # df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/labels_balanced_test_300.csv')
    df = df.rename(columns={'filename': 'jpg_file', 'label_name': 'gt'})
    df = df[df['gt'] != 'Other']

    l_jpg_file = []
    l_gt = []
    l_prediction = []
    l_description = []
    l_time = []
    l_class_time = []
    l_crop_ratio = []
    l_crop_values = []

    for i in tqdm(range(len(df))):
        jpg_file = df['jpg_file'].values[i]
        gt = df['gt'].values[i]
        start_time = time.time()

        if gt == 'SA-22':
            continue
        full_image_path = f"{TRAIN_FULL_MODE_FILES_PATH}{jpg_file}"
        model_output = get_list_of_points(client, full_image_path)
        create_molmo_crop_files(full_image_path, model_output, radius=100)
        num_of_objects = int((len(model_output) - 1) / 3)  # -1 - start from 1, /3 each object has 3 values
        start_cls_time = time.time()
        classifcation_result = classify_molmo_objects(client, TMP_FILES_FOLDER, num_of_objects)
        end_cls_time = time.time()
        classifcation_result = classifcation_result.replace("```json", "").replace("```", "").strip()
        classifcation_result = json.loads(classifcation_result)

        prediction = ""
        description = ""
        for i_res in range(num_of_objects):
            curr_result = classifcation_result['images'][i_res]
            prediction = curr_result['classification']
            prediction = update_prediction(prediction)
            description = curr_result['visual_evidence']
            break
        end_time = time.time()


        l_jpg_file.append(jpg_file)
        l_gt.append(gt)
        l_prediction.append(prediction)
        l_description.append(description)
        l_time.append(end_time - start_time)
        l_crop_ratio.append(200)
        l_crop_values.append(classifcation_result)
        l_class_time.append(end_cls_time - start_cls_time)


    df_tmp = pd.DataFrame({"jpg_file": l_jpg_file,
                           'gt': l_gt,
                           'prediction': l_prediction,
                           'description': l_description,
                           "total_diff_time": l_time,
                           'crop_ratio': l_crop_ratio,
                           'crop_values': l_crop_values,
                           "class_time": l_class_time})

    df_tmp.to_csv('150_molmo.csv', index=False)
    print_cm(df_tmp)

def test_molmo(full_image_path):
    client       = OpenAI(api_key="EMPTY", base_url=BASE_URL)
    model_output = get_list_of_points(client, full_image_path)
    create_molmo_crop_files(full_image_path, model_output, radius=100)
    #draw_with_molmo_bb(full_image_path, model_output)
    num_of_objects = int((len(model_output)-1)/3) # -1 - start from 1, /3 each object has 3 values
    classifcation_result = classify_molmo_objects(client, TMP_FILES_FOLDER, num_of_objects)
    classifcation_result = classifcation_result.replace("```json", "").replace("```", "").strip()
    classifcation_result = json.loads(classifcation_result)
    for curr_result in classifcation_result['images']:
        prediction = classifcation_result['images']['classification']
        prediction     = update_prediction(prediction)
        visual_evidence =  classifcation_result['images']['visual_evidence']
        print(prediction)



if __name__ == "__main__":
    full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_5/frame_444_00_11_806.jpg'
    #test_molmo(full_image_path)
    test_molmo_on_train()
    exit(0)

    DRAW_UPSAMPLE = False
    RUN_TRAIN_PIPELINE = False
    RUN_SINGLE_TEST = True
    RUN_IN_TEST_LOOP = False

    if RUN_TRAIN_PIPELINE:
        test_on_train()
        exit(0)

    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)
    # client = OpenAI(
    #     api_key="gpustack_a853c8a4cf87ee4b_6d6d0ade6d71fbadf5b015e04fb5e825",
    #     base_url="http://10.53.160.148/v1"
    # )

    if DRAW_UPSAMPLE:
        sim_img = simulate_vlm_view('/Testers/tmp_files/rec3_frame_273_00_06_418/crop_6.jpg')
        sim_img.show()

    if RUN_SINGLE_TEST:
        full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_2/frame_430_00_09_999.jpg'
        l_prediction, l_bb = run_pipeline(client, full_image_path)
        draw_box(full_image_path, l_bb, output_jpg_file=None, l_prediction=l_prediction, show_img=True)

    if RUN_IN_TEST_LOOP:
        OUTPUT_FOLDER = f'/home/amitli/repo/dor6_vision/Testers/tmp_results/'
        l_folders     = glob.glob('/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/*')
        l_folders.sort()
        l_time_diff       = []
        l_num_objs_in_img = []
        l_all_files       = []
        for folder in l_folders:

            base_folder_name = os.path.basename(folder)
            Path(f'{OUTPUT_FOLDER}{base_folder_name}').mkdir(parents=True, exist_ok=True)

            l_files = glob.glob(folder + '/*.jpg')
            for full_image_path in l_files:
                print(f"\t --- Process: {folder} {os.path.basename(full_image_path)} ---")
                start_time = time.time()
                l_prediction, l_bb = run_pipeline(client, full_image_path)
                end_time = time.time()

                l_time_diff.append(end_time - start_time)
                l_num_objs_in_img.append(len(l_bb))
                l_all_files.append(full_image_path)

                if len(l_bb) > 0:
                    # print(f"{os.path.basename(full_image_path)}")
                    output_jpg = f'{OUTPUT_FOLDER}{base_folder_name}/{os.path.basename(full_image_path)}'
                    draw_box(full_image_path, l_bb, output_jpg_file=output_jpg, l_prediction=l_prediction,
                             show_img=False)

        df_statisics = pd.DataFrame(
            {"jpg_file": l_all_files, "time_diff": l_time_diff, "num_objs_in_img": l_num_objs_in_img})
        df_statisics.to_csv(f'{OUTPUT_FOLDER}statistics.csv', index=False)
        plot_time_avg(df_statisics)
