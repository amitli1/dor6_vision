from Code_Train_B.run_model import get_results, draw_result
import pandas as pd
from PIL                              import Image, ImageDraw, ImageFont
import json

from app_config.settings import FONT_FILE


def eda_draw_result(df, full_path, filter_gt):

    jpg_file       = df['jpg_file'].values[0]
    gt_count       = df['gt_count'].values[0]
    l_gt_bb        = df['gt_bb'].values[0]
    l_gt_family    = df['gt_family'].values[0]
    l_model_family = df['model_family'].values[0]
    l_model_bb     = df['model_bb'].values[0]


    full_image_path = f'{full_path}/{jpg_file}'
    img        = Image.open(full_image_path)

    draw       = ImageDraw.Draw(img)
    img_w, img_h = img.size
    font = ImageFont.truetype(FONT_FILE, size=16)

    draw.text(
        (1, 1),
        f'{jpg_file}',
        fill="yellow",
        font=font
    )

    # add GT
    l_gt_family= []
    for i in range(len(l_gt_family)):
        label                     = l_gt_family[i]
        # if filter_gt != label:
        #     continue

        x_center, y_center, bw, bh = l_gt_bb[i]

        x_center = float(x_center)
        y_center = float(y_center)
        bw       = float(bw)
        bh       = float(bh)

        # Convert to pixel coordinates
        x_center *= img_w
        y_center *= img_h
        bw *= img_w
        bh *= img_h

        x1 = x_center - bw / 2
        y1 = y_center - bh / 2
        x2 = x_center + bw / 2
        y2 = y_center + bh / 2

        draw.rectangle([x1, y1, x2, y2], outline="green", width=1)
        draw.text((x1 + 50, y1), f"GT: {label}", fill="green", font=font)


    for i in range(len(l_model_bb)):
        ymin, xmin, ymax, xmax = l_model_bb[i]

        # 2. Convert from normalized (0-1000) to actual pixel values
        left = xmin * img_w / 1000
        top = ymin * img_h / 1000
        right = xmax * img_w / 1000
        bottom = ymax * img_h / 1000

        draw.rectangle([left, top, right, bottom], outline="red", width=1)
        label =  l_model_family[i].replace('none', 'Other')
        draw.text((left + 100, top),f"Pred: {label}", fill="red", font=font)

    img.show()


if __name__ == "__main__":

    jpg_files_path = '/home/amitli/datasets/DOR_6/Train_B/validation'
    pkl_result_file = f'/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_results.pkl'
    pkl_input_db_file = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_db.pkl'

    pkl_result_file = f'/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_results.pkl'

    # GT: Launchers, Pred: Anti aircraft, Jpg file: 1_524400_186_14-51-38.jpg
    # GT: Launchers, Pred: Anti aircraft, Jpg file: 1_564400_497_14-50-09.jpg
    # GT: Launchers, Pred: Anti aircraft, Jpg file: 1_524400_332_14-51-38.jpg
    # GT: Launchers, Pred: Anti aircraft, Jpg file: 1_484400_122_14-50-20.jpg

    file_name = '1_524400_186_14-51-38.jpg'
    df = pd.read_pickle(pkl_result_file)
    df_sample = df[df.jpg_file == file_name]

    eda_draw_result(df_sample, f"{jpg_files_path}/Images", filter_gt='Anti aircraft')
    exit(0)



    df_statistics_results = get_results(pkl_result_file, bb_threshold=0.4)

    for i in range(len(df_statistics_results)):
        jpg_file = df_statistics_results.jpg_file    .values[i]
        if jpg_file != '1_524400_186_14-51-38.jpg':
            continue
        pred     = df_statistics_results.model_family.values[i]
        gt       = df_statistics_results.gt_family   .values[i]

        if gt == 'Launchers' and pred == 'Anti aircraft':
            print(f'GT: {gt}, Pred: {pred}, Jpg file: {jpg_file}')
        # if gt == 'Anti aircraft' and pred != 'Uncertain':
        #     print(f'GT: {gt}, Pred: {pred}, Jpg file: {jpg_file}')



