from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import json
import cv2
import os

from pointing_agent.pointing_agent import PointingAgent

INPUT_CSV_FILE = '/home/amitli/repo/dor6_vision/weapon_system_classification/full_db_embedding.csv'
INPUT_DATA_DIR = r'/home/amitli/repo/VisionModels/input_images/dor_6_full_db/Train_A/data/Images/'
ZOOM_DATA_DIR = r'/home/amitli/repo/dor6_vision/Dataset/zoom_files/'
CROP_DATA_DIR = r'/home/amitli/repo/dor6_vision/Dataset/crop_files/'


def plot_zoom_image(image_path, x, y, output_crop_path, output_zoom_path):

    base_jpg_file = os.path.basename(image_path)
    image         = Image.open(image_path).convert("RGB")
    width, height = image.size

    crop_size     = 200
    left          = max(0, x - crop_size / 2)
    top           = max(0, y - crop_size / 2)
    right         = min(width, x + crop_size / 2)
    bottom        = min(height, y + crop_size / 2)

    crop_image  = image.crop((left, top, right, bottom))
    #crop_image.show()
    crop_image.save(f"{output_crop_path}/{base_jpg_file}", "JPEG")

    zoom_size  = 2

    zoom_img = crop_image.resize(
        (crop_image.width * zoom_size, crop_image.height * zoom_size),
        resample=Image.NEAREST  # or Image.BICUBIC for smoother
    )
    #zoom_img.show()
    zoom_img.save(f"{output_zoom_path}/{base_jpg_file}", "JPEG")

def get_all_db_points():
    pointingAgent = PointingAgent()

    df            = pd.read_csv(INPUT_CSV_FILE)
    df            = df[df['gt'] != 'Nothing']

    l_jpg_path   = []
    l_point_pred = []
    l_x          = []
    l_y          = []
    l_gt         = []

    for i in tqdm(range(len(df))):
        jpg_path         = rf"{INPUT_DATA_DIR}/{df.jpg_file.values[i]}"
        point_pred, x, y = pointingAgent.run_molmo_prediction(jpg_path)
        gt               = df['gt'].values[i]

        if x is not None:
            l_jpg_path   .append(jpg_path)
            l_point_pred .append(point_pred)
            l_x          .append(x)
            l_y          .append(y)
            l_gt         .append(gt)
        else:
            print(f"File: {jpg_path}, {gt} - cant point")

    df_res = pd.DataFrame({"jpg_file"  : l_jpg_path,
                           "gt"        : l_gt,
                           "point_pred": l_point_pred,
                           "x"         : l_x,
                           "y"         : l_y})

    df_res.to_csv("/home/amitli/repo/dor6_vision/Dataset/train.csv", index=False)

def create_zoom_files():
    df = pd.read_csv("/home/amitli/repo/dor6_vision/Dataset/train.csv")

    for i in tqdm(range(len(df))):
        jpg_file = df['jpg_file'].values[i]
        x        = df['x'].values[i]
        y        = df['y'].values[i]
        plot_zoom_image(jpg_file, x, y, CROP_DATA_DIR, ZOOM_DATA_DIR)

if __name__ == "__main__":

    #get_all_db_points()
    create_zoom_files()


