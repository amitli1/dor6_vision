from tqdm import tqdm
import pandas as pd
import glob
import os
import torch

import plotly.express as px
import matplotlib.pyplot as plt

from main_create_embeddings import create_embeddings, load_model, run_umap
from pointing_agent.pointing_agent import PointingAgent
from PIL import Image

def get_testset_pointing(test_set_path, output_name):
    pointingAgent = PointingAgent()

    l_jpg_files = glob.glob(f"{test_set_path}/*.jpg")

    l_jpg = []
    l_point_pred = []
    l_x = []
    l_y = []

    for full_file_path in tqdm(l_jpg_files):
        file                 = os.path.basename(full_file_path)
        generated_text, x, y = pointingAgent.run_molmo_prediction(full_file_path)

        l_jpg.append(file)
        l_point_pred.append(generated_text)
        l_x.append(x)
        l_y.append(y)

    df_res = pd.DataFrame({'jpg_file': l_jpg,"point_pred": l_point_pred, "x": l_x, "y": l_y})
    df_res.to_csv(output_name, index=False)

    return df_res


def crop_images(image_path, x, y, output_crop_path):

    base_name      = os.path.basename(image_path)
    image          = Image.open(image_path).convert("RGB")
    width, height  = image.size

    crop_size = 200
    left      = max(0, x - crop_size / 2)
    top       = max(0, y - crop_size / 2)
    right     = min(width, x + crop_size / 2)
    bottom    = min(height, y + crop_size / 2)

    crop_image = image.crop((left, top, right, bottom))
    # crop_image.show()
    crop_image.save(f"{output_crop_path}/{base_name}", "JPEG")

def crop_test_files(testset_with_point_csv, output_crop_path):

    df_point = pd.read_csv(testset_with_point_csv)
    for i in tqdm(range(len(df_point))):
        image_path = f"/home/amitli/repo/dor6_vision/Dataset/test_set/{df_point.jpg_file.values[i]}"
        crop_images(image_path, df_point.x.values[i], df_point.y.values[i], output_crop_path)

def test_set_crop_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_model(f"facebook/dinov2-base", device)
    create_embeddings(model, processor, device,
                      TEST_SET_POINT_CSV,
                      TEST_SET_CROP_FOLDER,
                      "/home/amitli/repo/dor6_vision/Dataset/embeddings_testset_crop.csv")

def run_umap_train_and_test_crop():
    df_train           = pd.read_csv("/home/amitli/repo/dor6_vision/Dataset/embeddings_train_crop.csv")
    df_test            = pd.read_csv("/home/amitli/repo/dor6_vision/Dataset/embeddings_testset_crop.csv")
    df_train_test_crop = pd.concat([df_train, df_test], ignore_index=True)
    df_train_test_crop.to_csv("/home/amitli/repo/dor6_vision/Dataset/train_test_crop_embeddings.csv")
    run_umap("/home/amitli/repo/dor6_vision/Dataset/train_test_crop_embeddings.csv",
             "/home/amitli/repo/dor6_vision/Dataset/train_test_umap_crop.csv")


def plot_umap_df(df_file_name):
    df        = pd.read_csv(df_file_name)
    color_map = {
        'SA-22'  : 'red',
        'SCUD'   : 'green',
        'T-90'   : 'blue',
        'unknown': 'black',
    }

    # Map colors to df['gt']
    colors = df['gt'].map(color_map)

    plt.figure(figsize=(8, 6))
    plt.scatter(df['umap_x'], df['umap_y'], c=colors, s=40, alpha=0.8)

    plt.xlabel('UMAP X')
    plt.ylabel('UMAP Y')
    plt.title(f'UMAP projection colored by GT ({len(df)} Samples)')
    plt.grid(True)

    # Add legend
    for cls, color in color_map.items():
        plt.scatter([], [], c=color, label=cls)
    plt.legend(title='GT')

    plt.show()


def plot_umap_to_html(df_file_name, plot_file_name):
    df          = pd.read_csv(df_file_name)
    df['hover'] = df["jpg_file"]
    fig = px.scatter(df, x='umap_x', y='umap_y', color='gt', hover_name="hover", title=f"#Samples: {len(df)}")

    fig.write_html(plot_file_name)
    fig.show()

if __name__ == "__main__":

    TRAIN_EMBEDDINGS_CROP_CSV = "/home/amitli/repo/dor6_vision/Dataset/embeddings_crop.csv"
    TEST_SET_PATH             = "/home/amitli/repo/dor6_vision/Dataset/test_set/"
    TEST_SET_CROP_FOLDER      = "/home/amitli/repo/dor6_vision/Dataset/test_set_crop/"
    TEST_SET_POINT_CSV        = "/home/amitli/repo/dor6_vision/Dataset/test_set_point.csv"

    #get_testset_pointing(TEST_SET_PATH, TEST_SET_POINT_CSV)
    #crop_test_files(TEST_SET_POINT_CSV, TEST_SET_CROP_FOLDER)
    #test_set_crop_embeddings()
    #run_umap_train_and_test_crop()

    plot_umap_df("/home/amitli/repo/dor6_vision/Dataset/train_test_umap_crop.csv")
    plot_umap_to_html("/home/amitli/repo/dor6_vision/Dataset/train_test_umap_crop.csv",
                      "/home/amitli/repo/dor6_vision/Dataset/train_test_crop.html")


