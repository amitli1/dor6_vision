from app_config.settings import FONT_FILE
from pointing_agent.pointing_agent import PointingAgent
from weapon_system_classification.weapon_system_classification import WeaponSystemClassification
from tqdm import tqdm
import glob
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import ast
import json

def get_text_nicely(image_path, text):

    jpg_file    = os.path.basename(image_path)
    text        = text.replace('np.float64', '')
    text        = ast.literal_eval(text)
    sorted_keys = sorted(text, key=lambda x: text[x], reverse=True)
    final       = f"{jpg_file}\n"
    for key in sorted_keys:
        final=final + f"{key} = {text[key]:.3f}\n"

    return final


def plot_img_with_point(image_path, x, y, text=""):

    img        = Image.open(image_path).convert("RGB")
    draw       = ImageDraw.Draw(img)
    r          = 5  # radius of the dot
    left_up    = (x - r, y - r)
    right_down = (x + r, y + r)

    if os.path.exists(FONT_FILE):
        font = ImageFont.truetype(FONT_FILE, size=24)
    else:
        font = ImageFont.load_default()

    draw.ellipse([left_up, right_down], fill="red", outline="red")
    draw.text((x + 15, y+15), get_text_nicely(image_path, text), fill="red", font= font)
    img.show()

if __name__ == '__main__':

    df = pd.read_csv("/home/amitli/repo/dor6_vision/results/test_set_crop_prediction.csv")

    i       = 100
    jpg_file = df["jpg_file"].values[i]
    full_file_path = f"/home/amitli/repo/dor6_vision/Dataset/test_set/{jpg_file}"
    point_x = df["point_x"].values[i]
    point_y = df["point_y"].values[i]
    classification_pred = df["classification_pred"].values[i]

    plot_img_with_point(full_file_path, point_x, point_y, classification_pred)
