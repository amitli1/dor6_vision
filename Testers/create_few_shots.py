from PIL                             import Image, ImageDraw, ImageFont
from openai                          import OpenAI
from Testers.Pipeline_Crop_And_Classify import get_list_of_bounding_boxes
from app_config.settings import FONT_FILE

import json
import os

def draw_result(image_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_FILE, size=32)
    width, height = img.size
    draw.text((1, 1), f"{width}x{height}", fill="red", font=font)
    img.show()


def create_crop_files(input_image_path, model_bb_json_res, min_crop_size, output_path):

    image         = Image.open(input_image_path).convert("RGB")
    width, height = image.size
    l_crop_ratio  = []

    for i, bb in enumerate(model_bb_json_res):
        ymin, xmin, ymax, xmax = bb['box_2d']
        left       = (xmin / 1000) * width
        top        = (ymin / 1000) * height
        right      = (xmax / 1000) * width
        bottom     = (ymax / 1000) * height

        crop_width  = int(right-left)
        crop_height = int(bottom-top)
        crop_size   = (crop_width * crop_height)/(width * height)
        l_crop_ratio.append(crop_size)

        if crop_width < min_crop_size:
            delta  = (min_crop_size - crop_width) / 2
            left  -= delta
            right += delta

        if crop_height < min_crop_size:
            delta = (min_crop_size - crop_height) / 2
            top    -= delta
            bottom += delta

        crop_image = image.crop((left, top, right, bottom))
        crop_image.save(output_path, "JPEG")

    return l_crop_ratio

if __name__ == "__main__":

    sa_22_1 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-44_444400_23.jpg'
    sa_22_2 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-20-27_844400_795.jpg'
    sa_22_3 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-21-02_1244400_1020.jpg'

    scud_1 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-17-54_524400_136.jpg'
    scud_2 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-20-34_884400_1224.jpg'
    scud_3 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-21-10_1324400_673.jpg'

    t90_1 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-18-04_484400_633.jpg'
    t90_2 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-20-40_924400_731.jpg'
    t90_3 = '/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/11-21-17_1284400_311.jpg'

    l_files = [sa_22_1 , sa_22_2 , sa_22_3,
               scud_1  , scud_2  , scud_3,
               t90_1   , t90_2   , t90_3]

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:9000/v1")

    l_gt = ['SA-22'] * 3 + ['SCUD'] * 3 + ['T-90'] * 3
    for i, gt in enumerate(l_gt):
        output_path = '/home/amitli/repo/dor6_vision/Testers/tmp_few_shots/'
        output_file = f"{output_path}{gt}_{i%3+1}.JPG"
        input_file  = l_files[i]
        model_json_res = get_list_of_bounding_boxes(client, input_file)
        create_crop_files(input_file, model_json_res, 128, output_file)


    draw_result('/home/amitli/repo/dor6_vision/Testers/tmp_few_shots/SA-22_1.JPG')