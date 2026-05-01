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


def run_gemma4(file_name):

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")
    messages = get_all_bb(file_name)
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

def draw_box(image_path, l_cords, output_jpg_file=None, show_img=False):
    # Load the image
    img           = Image.open(image_path)
    draw          = ImageDraw.Draw(img)
    width, height = img.size

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

    if output_jpg_file:
        crop_image = img.crop((left, top, right, bottom))
        crop_image.save(output_jpg_file, "JPEG")
    if show_img:
        img.show()

def plot_few_shots():
    import matplotlib.pyplot as plt
    from PIL import Image
    import glob


    l_image_paths = glob.glob('/home/amitli/repo/dor6_vision/Testers/few_shots/*.JPG')
    l_image_paths.sort()

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        img = Image.open(l_image_paths[i])
        title = f"{os.path.basename(l_image_paths[i])} [{img.size}]"
        #ax.imshow(img, aspect='auto')  # stretch image
        ax.imshow(img)  # stretch image
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def eda_text():
    scud_txt_1 = "A heavy multi‑axle military missile transporter‑erector‑launcher carrying a long cylindrical ballistic missile, painted olive green with a tan missile, viewed from above at an angle on a paved road, rendered in a realistic military simulation style"
    scud_txt_2 = "Side view of a long olive‑green missile transporter‑erector‑launcher vehicle with multiple axles, carrying a horizontally mounted cylindrical missile, rendered in a realistic military simulation environment on a paved road"
    scud_txt_3 = "A tall ballistic missile standing vertically on a military launcher platform, tan missile body with a pointed nose cone, mounted on a green rectangular erector base, viewed from an elevated angle in a realistic military simulation environment."

    sa22_txt_1 = "Wheeled self‑propelled air‑defense vehicle with a multi‑axle truck chassis, olive‑green camouflage, roof‑mounted missile canisters and radar sensors on a raised turret, shown from an elevated angle in a military simulation environment."
    sa22_txt_2 = "Top‑down view of a wheeled air‑defense combat vehicle with an armored cab, a raised turret carrying rectangular missile canisters and sensor housings, olive‑green military color, rendered in a realistic simulation environment on a paved road."
    sa22_txt_3 = "Top‑down view of a wheeled air‑defense military vehicle with an armored cab, raised rear turret, and multiple long cylindrical weapon elements mounted lengthwise, olive‑green color, rendered in a realistic simulation environment on a paved road."

    t90_txt_1 = "Top‑down view of an olive‑green tracked main battle tank with a central rotating turret and long forward‑facing gun barrel, rendered in a realistic military simulation environment on a paved road."
    t90_txt_2 = "Elevated oblique view of an olive‑green tracked main battle tank with a central turret and long forward‑facing gun barrel, driving on a paved road, rendered in a realistic military simulation environment."
    t90_txt_3 = "Oblique overhead view of an olive‑green tracked main battle tank with a central rounded turret and a long forward‑facing gun barrel, rendered in a realistic military simulation environment on a paved surface."

    other_txt_1 = ( "The image shows no guns and no radar, so it can't be an SA-22. "
                    "It contains multiple compact canisters, so it can't be a Scud (which uses a single large missile). "
                    "The image shows a naval launcher, not a tracked armored vehicle, so it can't be a T-90.")

if __name__ == "__main__":

    # this is crop from simulation image. please describe the weapon system in this image. (I want to take your description and use it as few shot for prompt)
    PLOT_FEW_SHOTS = True
    CREATE_CROPS = False

    if CREATE_CROPS:
        file_sa_22_1 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-44_444400_19.jpg"
        file_sa_22_2 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-44_444400_117.jpg"
        file_sa_22_3 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-44_444400_181.jpg"

        file_scud_1 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-54_524400_2.jpg"
        file_scud_2 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-54_524400_145.jpg"
        file_scud_3 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-54_524400_613.jpg"

        file_t90_1 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-18-04_484400_215.jpg"
        file_t90_2 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-18-04_484400_614.jpg"
        file_t90_3 = r"/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-21-17_1284400_246.jpg"

        l_sa22  = [file_sa_22_1, file_sa_22_2, file_sa_22_3]
        l_scud  = [file_scud_1, file_scud_2, file_scud_3]
        l_t90   = [file_t90_1, file_t90_2, file_t90_3]

        for i in range(len(l_sa22)):
            file      = l_sa22[i]
            res_json  = run_gemma4(file)
            gemma_res = res_json[0]['box_2d']
            draw_box(file,
                     [gemma_res],
                     output_jpg_file=f'/home/amitli/repo/dor6_vision/Testers/few_shots/SA-22_{i+1}.JPG',
                     show_img=False)


        for i in range(len(l_scud)):
            file      = l_scud[i]
            res_json  = run_gemma4(file)
            gemma_res = res_json[0]['box_2d']
            draw_box(file,
                     [gemma_res],
                     output_jpg_file=f'/home/amitli/repo/dor6_vision/Testers/few_shots/SCUD_{i+1}.JPG',
                     show_img=False)

        for i in range(len(l_t90)):
            file      = l_t90[i]
            res_json  = run_gemma4(file)
            gemma_res = res_json[0]['box_2d']
            draw_box(file,
                     [gemma_res],
                     output_jpg_file=f'/home/amitli/repo/dor6_vision/Testers/few_shots/T-90_{i+1}.JPG',
                     show_img=False)

    if PLOT_FEW_SHOTS:
        plot_few_shots()


