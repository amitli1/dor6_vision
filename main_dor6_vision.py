from app_config.settings import FONT_FILE, TEST_FULL_MODE_FILES_PATH, TEST_CROP_WITH_PREDICTION_PATH, \
    TEST_CROP_WITH_PREDICTION_HTML
from pointing_agent.pointing_agent import PointingAgent
from weapon_system_classification.weapon_system_classification import WeaponSystemClassification
from tqdm import tqdm
import glob
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import ast
import json
import numpy as np
from pathlib import Path

def get_text_nicely(image_path, text):

    jpg_file    = os.path.basename(image_path)
    text        = text.replace('np.float64', '')
    text        = ast.literal_eval(text)
    sorted_keys = sorted(text, key=lambda x: text[x], reverse=True)
    final       = f"{jpg_file}\n"
    for key in sorted_keys:
        final=final + f"{key} = {text[key]:.3f}\n"

    return final


def plot_img_with_point(image_path, x, y, text="", output_path=None):

    img        = Image.open(image_path).convert("RGB")
    draw       = ImageDraw.Draw(img)
    r          = 5  # radius of the dot
    left_up    = (x - r, y - r)
    right_down = (x + r, y + r)

    if os.path.exists(FONT_FILE):
        font = ImageFont.truetype(FONT_FILE, size=24)
    else:
        font = ImageFont.load_default()

    if np.isnan(x) == False:
        draw.ellipse([left_up, right_down], fill="red", outline="red")
        draw.text((x + 15, y+15), get_text_nicely(image_path, text), fill="red", font= font)
    #img.show()
    if output_path is not None:
        img.save(output_path)

def create_jpg_prediction(df):

    for i in tqdm(range(len(df))):
        jpg_file            = df["jpg_file"].values[i]
        full_file_path      = f"{TEST_FULL_MODE_FILES_PATH}/{jpg_file}"
        point_x             = df["point_x"].values[i]
        point_y             = df["point_y"].values[i]
        classification_pred = df["classification_pred"].values[i]
        output_path         = f"{TEST_CROP_WITH_PREDICTION_PATH}/{jpg_file}"
        plot_img_with_point(full_file_path, point_x, point_y, classification_pred, output_path)

def create_html_prediction():
    image_folder = TEST_CROP_WITH_PREDICTION_PATH
    output_html = TEST_CROP_WITH_PREDICTION_HTML

    # Collect all jpg files
    image_paths = sorted(Path(image_folder).glob("*.jpg"))

    # Start HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Image Gallery</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(1, 1fr); /* 1 columns */
                gap: 20px;
            }

            .item {
                text-align: center;
            }

            img {
                width: 100%;
                max-width: 250px;
                height: auto;
                display: block;
                margin: 0 auto;
            }

            .title {
                margin-top: 5px;
                font-size: 14px;
                word-break: break-all;
            }
        </style>
    </head>
    <body>

    <h2>Image Gallery</h2>

    <div class="grid">
    """

    # Add images
    for img_path in image_paths:
        filename = img_path.name
        title = os.path.splitext(filename)[0]

        html += f"""
        <div class="item">
            <img src="{image_folder}/{filename}" alt="{title}" loading="lazy">
            <div class="title">{title}</div>
        </div>
        """

    # Close HTML
    html += """
    </div>

    </body>
    </html>
    """

    # Save file
    with open(output_html, "w") as f:
        f.write(html)

    print(f"HTML gallery saved to {output_html}")

if __name__ == '__main__':

    # frame 2040 (tank > sa22 wrong)

    df = pd.read_csv("/home/amitli/repo/dor6_vision/results/test_set_crop_prediction.csv")
    df_umap = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/train_test_umap_crop.csv')
    # create_jpg_prediction(df)
    # create_html_prediction()
    # exit(0)

    i       = 50
    jpg_file = df["jpg_file"].values[i]

    jpg_file = "frame_1170.jpg"
    df_tmp   = df[df["jpg_file"] == jpg_file]



    full_file_path = f"{TEST_FULL_MODE_FILES_PATH}/{jpg_file}"
    point_x = df_tmp["point_x"].values[0]
    point_y = df_tmp["point_y"].values[0]
    classification_pred = df_tmp["classification_pred"].values[0]

    plot_img_with_point(full_file_path, point_x, point_y, classification_pred)
    df_umap_file = df_umap[df_umap.jpg_file == jpg_file]

    print(f"{df_umap_file.jpg_file.values[0]}, {df_umap_file.umap_x.values[0]}, {df_umap_file.umap_y.values[0]} tag = {df_umap_file['gt'].values[0]} pred ={classification_pred}")
