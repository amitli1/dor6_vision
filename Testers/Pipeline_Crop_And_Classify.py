from openai                          import OpenAI
from PIL                             import Image, ImageDraw, ImageFont
from Prompt.create_classifier_prompt import get_all_bb
from app_config.settings             import FONT_FILE, TRAIN_FULL_MODE_FILES_PATH
from main_classification_with_vlm    import print_cm
from tqdm                            import tqdm

import pandas                        as pd
import numpy                         as np

import ast
import time
import json
import os
import glob
import base64

TMP_FILES_FOLDER = '/home/amitli/repo/dor6_vision/Testers/tmp_files'
BB_TMP_FILE      = "/home/amitli/repo/dor6_vision/Testers/tmp_files/bb.json"

def draw_box(image_path, l_cords, output_jpg_file=None, l_prediction=None, show_img=False):
    # Load the image
    img           = Image.open(image_path)
    draw          = ImageDraw.Draw(img)
    width, height = img.size
    font          = ImageFont.truetype(FONT_FILE, size=32)

    pred_str = ""
    for i in range(len(l_prediction)):
        pred_str += l_prediction[i] + "\n"
    draw.text((1, 1), pred_str, fill="red", font=font)


    if type(l_cords) == str:
        l_cords = ast.literal_eval(l_cords)

    for i, coords in enumerate(l_cords):
        ymin, xmin, ymax, xmax = coords

        # 2. Convert from normalized (0-1000) to actual pixel values
        left   = xmin * width / 1000
        top    = ymin * height / 1000
        right  = xmax * width / 1000
        bottom = ymax * height / 1000

        # 3. Draw the rectangle
        # PIL expects [xmin, ymin, xmax, ymax]
        #draw.rectangle([left, top, right, bottom], outline="red", width=3)
        # if l_prediction is not None:
        #     draw.text((left+50, top), l_prediction[i], fill="red", font=font)
        draw.text((left+50, top), f"{i+1}", fill="red", font=font)

    if output_jpg_file:
        img.save(output_jpg_file)
    if show_img:
        img.show()

def get_list_of_bounding_boxes(client, full_image_path):
    messages = get_all_bb(full_image_path)
    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        extra_body={
            "mm_processor_kwargs": {
                "max_soft_tokens": 1120
            }
        }
    )
    res_text = response.choices[0].message.content
    res_text = res_text.replace("```json", "").replace("```", "").strip()
    res_json = json.loads(res_text)
    return res_json

def create_crop_files(image_path, model_bb_json_res):

    image         = Image.open(image_path).convert("RGB")
    width, height = image.size
    #draw          = ImageDraw.Draw(image)

    for i, bb in enumerate(model_bb_json_res):
        ymin, xmin, ymax, xmax = bb['box_2d']
        left       = (xmin / 1000) * width
        top        = (ymin / 1000) * height
        right      = (xmax / 1000) * width
        bottom     = (ymax / 1000) * height
        #draw.rectangle([left, top, right, bottom], outline="red", width=3)
        crop_image = image.crop((left, top, right, bottom))
        #crop_image.show()
        crop_image.save(f"{TMP_FILES_FOLDER}/crop_{i+1}.jpg", "JPEG")

    #image.show()

def classify_objects(client, objects_path, num_of_objects):

    #messages = create_prompt_classification_for_crops(objects_path, num_of_objects)
    messages = get_classification_prompt(objects_path, num_of_objects)

    schema = {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "classification": {"type": "string"}
                    },
                    "required": ["description", "classification"]
                }
            }
        },
        "required": ["images"]
    }


    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        extra_body={
            "guided_json": schema,
            "mm_processor_kwargs": {
                "max_soft_tokens": 280 #140
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
    content.append({"type": "image_url", "image_url": {"url":f"data:image/jpeg;base64,{encode_image(image_path)}"}})


def get_classification_prompt(objects_path, num_of_objects):

    FEW_SHOTS_FOLDER = '/home/amitli/repo/dor6_vision/Testers/few_shots'

    sa22_txt_1 = "Wheeled self‑propelled air‑defense vehicle with a multi‑axle truck chassis, olive‑green camouflage, roof‑mounted missile canisters and radar sensors on a raised turret, shown from an elevated angle in a military simulation environment."
    sa22_txt_2 = "Top‑down view of a wheeled air‑defense combat vehicle with an armored cab, a raised turret carrying rectangular missile canisters and sensor housings, olive‑green military color, rendered in a realistic simulation environment on a paved road."
    sa22_txt_3 = "Top‑down view of a wheeled air‑defense military vehicle with an armored cab, raised rear turret, and multiple long cylindrical weapon elements mounted lengthwise, olive‑green color, rendered in a realistic simulation environment on a paved road."

    scud_txt_1 = "A heavy multi‑axle military missile transporter‑erector‑launcher carrying a long cylindrical ballistic missile, painted olive green with a tan missile, viewed from above at an angle on a paved road, rendered in a realistic military simulation style"
    scud_txt_2 = "Side view of a long olive‑green missile transporter‑erector‑launcher vehicle with multiple axles, carrying a horizontally mounted cylindrical missile, rendered in a realistic military simulation environment on a paved road"
    scud_txt_3 = "A tall ballistic missile standing vertically on a military launcher platform, tan missile body with a pointed nose cone, mounted on a green rectangular erector base, viewed from an elevated angle in a realistic military simulation environment."

    t90_txt_1 = "Top‑down view of an olive‑green tracked main battle tank with a central rotating turret and long forward‑facing gun barrel, rendered in a realistic military simulation environment on a paved road."
    t90_txt_2 = "Elevated oblique view of an olive‑green tracked main battle tank with a central turret and long forward‑facing gun barrel, driving on a paved road, rendered in a realistic military simulation environment."
    t90_txt_3 = "Oblique overhead view of an olive‑green tracked main battle tank with a central rounded turret and a long forward‑facing gun barrel, rendered in a realistic military simulation environment on a paved surface."


    content = []
    add_text_line(content, f"You receive {num_of_objects} patches of images from a military simulation and you must classify the military vehicle in the image, based only on the vehicle structure and mounted weapon system. Ignore background, terrain, and camera angle.")
    add_text_line(content,"Classify the military vehicle in each image. If the military vehicle is small or distant, consider its overall shape, color patterns.")

    add_text_line(content, "VISUAL DIFFERENTIATORS")
    add_text_line(content, "SA-22")
    add_text_line(content,"1. A truck‑mounted system")
    add_text_line(content,"2. Has a visible dual autocannons")
    add_text_line(content,"3. Has a radar module")
    add_text_line(content,"4. The missiles are mounted on the sides of the turret, not in the center.")

    add_text_line(content, "SCUD")
    add_text_line(content, "1. carry one large ballistic missile")
    add_text_line(content, "2. Very long TEL truck")
    add_text_line(content, "3. No radar antennas")

    add_text_line(content, "T-90")
    add_text_line(content, "1. TANK")
    add_text_line(content, "2. Large gun turret with a single main cannon")
    add_text_line(content, "3. Prominent rotating turret")
    #add_text_line(content, "4. Tracks instead of wheels")

    add_text_line(content, "Reject SA-22 if:")
    add_text_line(content, "1. The missiles are NOT mounted on the sides of the turret")
    add_text_line(content, "2. Contains one missile")



    add_text_line(content,"Examples:")

    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SA-22_1.JPG')
    add_text_line(content, f"{sa22_txt_1} Answer: SA-22")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SA-22_2.JPG')
    add_text_line(content, f"{sa22_txt_2} Answer: SA-22")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SA-22_3.JPG')
    add_text_line(content, f"{sa22_txt_3} Answer: SA-22")

    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SCUD_1.JPG')
    add_text_line(content, f"{scud_txt_1} Answer: SCUD")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SCUD_2.JPG')
    add_text_line(content, f"{scud_txt_2} Answer: SCUD")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/SCUD_3.JPG')
    add_text_line(content, f"{scud_txt_3} Answer: SCUD")

    add_image_line(content, f'{FEW_SHOTS_FOLDER}/T-90_1.JPG')
    add_text_line(content, f"{t90_txt_1} Answer: T-90")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/T-90_2.JPG')
    add_text_line(content, f"{t90_txt_2} Answer: T-90")
    add_image_line(content, f'{FEW_SHOTS_FOLDER}/T-90_3.JPG')
    add_text_line(content, f"{t90_txt_3} Answer: T-90")

    add_text_line(content, 'If an image is too blurry to identify, label it as "Uncertain"')
    add_text_line(content, "Based on the examples above, which class does the following images belong to? If the image does not fit any of the three, answer 'none'. Answer only: 'SA-22', 'SCUD', 'T-90', or 'none' or 'Uncertain'.")
    #add_text_line(content,"for each image, describe the shape and color, then provide the classification")
    add_text_line(content, "for each image, return a JSON object with fields: description, classification")
    for i in range(num_of_objects):
        add_text_line(content, f"Image: {i+1}")
        crop_file_path = f"{objects_path}/crop_{i + 1}.jpg"
        add_image_line(content, crop_file_path)

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


def create_prompt_classification_for_crops(objects_path, num_of_objects):
    base_prompt = f"""
        Classify these {num_of_objects} simulation images patches in order:
    """


    base_prompt_continue = """         
        The image may contain 'SA-22', 'SCUD', 'T-90' or 'military_vehicle'                
    """

    rule_prompt = """           
              - Use one of the following labels: 'SA-22', 'SCUD', 'T-90', or 'military_vehicle'                                    
              - If the vehicle is not one of the first three, use 'military_vehicle'
              - Base decisions on visible shape and structure
              """

    vehicle_properties = """
        [CRITICAL VISUAL DIFFERENTIATORS]
       1.   SCUD Missile Launcher (TEL – Transporter Erector Launcher):
            * a long transporter-erector-launcher carrying a single large missile.
    
        2.  SA-22 (Pantsir-S1) Air Defense System:
            * mobile surface-to-air missile system with distinctive radar panels and multiple missile launch tubes mounted on a truck chassis.
    
        3. T-90 Main Battle Tank:
            * a compact tracked chassis with a turret, consistent with a main battle tank rather than a missile launcher system.
            * does not show the long missile body of radar/missile
            * has a distinctive turret
            """

    content = []
    add_text_line(content, base_prompt)
    for i in range(num_of_objects):
        crop_file_path = f"{objects_path}/crop_{i+1}.jpg"
        add_image_line(content, crop_file_path)
    add_text_line(content, base_prompt_continue)
    add_text_line(content, rule_prompt)
    #add_text_line(content, vehicle_properties)

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

def test_on_train():

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    #df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/labels_balanced_test_500.csv')
    df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/shiry_testset_balanced.csv')
    df = df.rename(columns={'filename': 'jpg_file', 'label_name': 'gt'})
    df = df[df['gt'] != 'Other']

    l_jpg_file    = []
    l_gt          = []
    l_prediction  = []
    l_description = []
    l_time        = []

    for i in tqdm(range(len(df))):
        jpg_file        = df['jpg_file'].values[i]
        gt              = df['gt'].values[i]

        try:
            full_image_path = f"{TRAIN_FULL_MODE_FILES_PATH}{jpg_file}"
            start_time      = time.time()
            model_json_res = get_list_of_bounding_boxes(client, full_image_path)
            with open(BB_TMP_FILE, "w") as f:
                json.dump(model_json_res, f, indent=4)
            create_crop_files(full_image_path, model_json_res)
            classifcation_result = classify_objects(client, TMP_FILES_FOLDER, len(model_json_res))
            classifcation_result = classifcation_result.replace("```json", "").replace("```", "").strip()
            classifcation_result = json.loads(classifcation_result)
            if type(classifcation_result) == list:
                classifcation_result = classifcation_result[0]
            prediction           = classifcation_result["classification"]
            description          = classifcation_result["description"]
        except Exception as e:
            print(f"Error in {jpg_file}")
            prediction  = "Error"
            description = "Error"

        end_time = time.time()
        l_jpg_file    .append(jpg_file)
        l_gt          .append(gt)
        l_prediction  .append(prediction)
        l_description .append(description)
        l_time        .append(end_time-start_time)

    df_tmp = pd.DataFrame({"jpg_file": l_jpg_file, 'gt': l_gt, 'prediction': l_prediction, 'description': l_description, "diff_time": l_time})
    df_tmp.to_csv('tmp.csv', index=False)
    print_cm(df_tmp)

if __name__ == "__main__":

    RUN_TRAIN_PIPELINE = True

    if RUN_TRAIN_PIPELINE:
        test_on_train()
        # df_tmp = pd.read_csv('tmp.csv')
        # print_cm(df_tmp)
        exit(0)

    client          = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    #x full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_3/frame_273_00_06_418.jpg'
    #full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_2/frame_344_00_07_999.jpg'
    #full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_2/frame_473_00_10_999.jpg'
    full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_1/frame_247_00_06_363.jpg'
    # x full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_4/frame_287_00_06_944.jpg'
    # x full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_4/frame_328_00_07_936.jpg'
    #full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_5/frame_481_00_12_790.jpg'
    #full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_5/frame_629_00_16_725.jpg'
    # x full_image_path = '/home/amitli/repo/dor6_vision/Dataset/test_set_v2/jpgs/VBS_Record_5/frame_259_00_06_887.jpg'




    #draw_box(full_image_path, l_cords=[], output_jpg_file=None, show_img=True)

    DRAW_UPSAMPLE     = False
    GET_LIST_OF_BB    = True
    CREATE_CROP_FILES = True
    CLASSIFY          = True

    if DRAW_UPSAMPLE:
        sim_img = simulate_vlm_view('/Testers/tmp_files/rec3_frame_273_00_06_418/crop_6.jpg')
        sim_img.show()
    if GET_LIST_OF_BB:
        start_time = time.time()
        model_json_res = get_list_of_bounding_boxes(client, full_image_path)
        end_time = time.time()
        with open(BB_TMP_FILE, "w") as f:
            json.dump(model_json_res, f, indent=4)
        print(f"[{(end_time-start_time):.2f} sec] Get list of BB")
    if CREATE_CROP_FILES:
        with open(BB_TMP_FILE, "r") as f:
            model_json_res = json.load(f)
        start_time = time.time()
        create_crop_files(full_image_path, model_json_res)
        end_time = time.time()
        print(f"[{(end_time - start_time):.2f} sec] Create crop files")
    if CLASSIFY:
        with open(BB_TMP_FILE, "r") as f:
            model_json_res = json.load(f)
        start_time = time.time()
        classifcation_result = classify_objects(client, TMP_FILES_FOLDER, len(model_json_res))
        classifcation_result = classifcation_result.replace("```json", "").replace("```", "").strip()
        classifcation_result = json.loads(classifcation_result)
        end_time = time.time()

        l_prediction         = []
        l_bb                 = []
        for i in range(len(model_json_res)):
            pred           = classifcation_result[i]
            classification = pred["classification"]
            description    = pred["description"]

            l_prediction.append(f"[{i+1}] {classification}")
            l_bb        .append(model_json_res[i]['box_2d'])
            print(f'\t[{i+1}] [{classification}] {description}')

        print(f"[{(end_time - start_time):.2f} sec] Object classification")
        draw_box(full_image_path, l_bb, output_jpg_file=None, l_prediction=l_prediction, show_img=True)






