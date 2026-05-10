import pandas as pd
from PIL                             import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
from app_config.settings import TRAIN_FULL_MODE_FILES_PATH
import os

def get_list_of_close_right_jpgs(df_right, wrong_jpg_file, files_path):
    wrong_prefix     = wrong_jpg_file.rsplit("_", 1)[0]
    l_right_files    = df_right.jpg_file.values
    l_right_prefixes = [f.rsplit("_", 1)[0] for f in l_right_files]
    l_res_files      = []
    l_res_desc       = []
    for i in range(len(l_right_prefixes)):
        if wrong_prefix == l_right_prefixes[i]:
            l_res_files.append(f"{files_path}{l_right_files[i]}")
            l_res_desc.append(df_right.description.values[i])

    return l_res_files, l_res_desc

def plot_close_true_files(wrong_jpg_file, l_close_true_files):
    # number of images on the right
    # ----- layout -----
    n_true = len(l_close_true_files)

    # right side: 3 columns
    right_cols = 3
    right_rows = math.ceil(n_true / right_cols)

    # total grid:
    # left column for wrong image + 3 columns for true images
    total_cols = 1 + right_cols

    fig = plt.figure(figsize=(4 * total_cols, 4 * right_rows))

    # LEFT image spans all rows
    ax_left = plt.subplot2grid(
        (right_rows, total_cols),
        (0, 0),
        rowspan=right_rows
    )

    img_wrong = Image.open(wrong_jpg_file)
    ax_left.imshow(img_wrong)
    ax_left.set_title(f"{os.path.basename(wrong_jpg_file)}")
    ax_left.axis("off")

    # RIGHT images in 3-column grid
    for idx, true_path in enumerate(l_close_true_files):
        row = idx // right_cols
        col = (idx % right_cols) + 1  # +1 because col 0 is the wrong image

        ax = plt.subplot2grid(
            (right_rows, total_cols),
            (row, col)
        )

        img_true = Image.open(true_path)
        ax.imshow(img_true)
        ax.set_title(f"{os.path.basename(true_path)}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def handle_wrong(df_right, df_wrong, index):
    wrong_jpg_file  = df_wrong.jpg_file.values[index]
    full_image_path = f"{TRAIN_FULL_MODE_FILES_PATH}{wrong_jpg_file}"
    gt              = df_wrong['gt'].values[index]
    prediction      = df_wrong['prediction'].values[index]
    dsecription =    df_wrong['description'].values[index]

    l_right_files, l_res_desc = get_list_of_close_right_jpgs(df_right, wrong_jpg_file,TRAIN_FULL_MODE_FILES_PATH)
    l_right_files = l_right_files[:6]
    l_res_desc    = l_res_desc[:6]
    for desc in l_res_desc:
        print(desc)

    #img = Image.open(full_image_path)
    #img.show()

    print(wrong_jpg_file)
    print(gt)
    print(prediction)
    print("----")
    print(dsecription)
    print("----")

    plot_close_true_files(full_image_path, l_right_files)

if __name__ == "__main__":
    df       = pd.read_csv('tmp.csv')
    df_sa22  = df[df['gt'] == 'SA-22']
    df_scud = df[df['gt'] == 'SCUD']
    df_t90 = df[df['gt'] == 'T-90']

    handle_wrong(df_scud[df_scud['gt'] == df_scud['prediction']],
                 df_scud[df_scud['gt'] != df_scud['prediction']],
                 1)
