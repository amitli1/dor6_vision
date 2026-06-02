from tqdm                             import tqdm
from Code_Train_B.vlm_model           import VlmModel
from PIL                              import Image, ImageDraw, ImageFont
import pandas as pd

import json

from app_config.settings import FONT_FILE


class ModelResult:
    def __init__(self):
        self.l_jpg_file          = []
        self.l_gt_count          = []
        self.l_gt_bb             = []
        self.l_gt_family         = []
        self.l_model_bb          = []
        self.l_model_family      = []
        self.l_model_description = []

    def add(self, jpg_file, l_model_bb, l_model_family, l_model_description):

        l_gt_bb     = df[df.jpg_file == jpg_file].bb.values[0]
        l_gt_family = df[df.jpg_file == jpg_file].family.values[0]

        self.l_jpg_file          .append(jpg_file)
        self.l_gt_count          .append(len(l_gt_bb))
        self.l_gt_bb             .append(l_gt_bb)
        self.l_gt_family         .append(l_gt_family)
        self.l_model_bb          .append(l_model_bb)
        self.l_model_family      .append(l_model_family)
        self.l_model_description .append(l_model_description)

    def save_to_pickle(self, pkl_name):
        df = pd.DataFrame({'jpg_file'          : self.l_jpg_file,
                           'gt_count'          : self.l_gt_count,
                           'gt_bb'             : self.l_gt_bb,
                           'gt_family'         : self.l_gt_family,
                           'model_bb'          : self.l_model_bb,
                           'model_family'      : self.l_model_family,
                           'model_description' : self.l_model_description})
        df.to_pickle(pkl_name)

def draw_result(df, full_path):

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
    for i in range(len(l_gt_family)):
        label                     = l_gt_family[i]
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

        #draw.rectangle([x1, y1, x2, y2], outline="green", width=1)
        draw.text((x1 + 50, y1), label, fill="green", font=font)


    for i in range(len(l_model_bb)):
        ymin, xmin, ymax, xmax = l_model_bb[i]

        # 2. Convert from normalized (0-1000) to actual pixel values
        left = xmin * img_w / 1000
        top = ymin * img_h / 1000
        right = xmax * img_w / 1000
        bottom = ymax * img_h / 1000

        draw.rectangle([left, top, right, bottom], outline="red", width=1)
        draw.text((left + 100, top), l_model_family[i].replace('none', 'Other'), fill="red", font=font)

    img.show()


if __name__ == "__main__":


    RUN_MODEL = False
    DB_TYPE   = "Train"

    if DB_TYPE == "Train":
        jpg_files_path  = '/home/amitli/datasets/DOR_6/Train_B/Database'
        pkl_result_file = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles//train_results.pkl'
        pkl_db_file     = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/train_db.pkl'
    else:
        jpg_files_path = '/home/amitli/datasets/DOR_6/Train_B/validation'
        pkl_result_file = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_results.pkl'
        pkl_db_file     = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_db.pkl'

    if RUN_MODEL is False:
        df = pd.read_pickle(pkl_result_file)
        for i in range(len(df)):
            df_current = df[df.jpg_file == df.jpg_file[i]]
            draw_result(df_current, f'{jpg_files_path}/Images')
        exit(0)


    df                      = pd.read_pickle(pkl_db_file)
    df                      = df[df['num_gt'] > 0]
    df                      = df.sample(n=20)
    modelResult             = ModelResult()

    vlmModel = VlmModel()
    for i in tqdm(range(len(df))):
        jpg_file       = df.jpg_file.values[i]

        # if jpg_file != '1_564400_8_14-50-09.jpg':
        #     continue

        l_gt_targets   = df.targets.values[i]
        l_gt_bb        = df.bb.values[i]
        full_jpg_file  = f'{jpg_files_path}/Images/{jpg_file}'
        l_model_bb_res = vlmModel.get_list_of_bounding_boxes(full_jpg_file)
        l_crop_ratio   = vlmModel.create_crop_files(full_jpg_file, l_model_bb_res, min_crop_size=128)
        family_result  = vlmModel.classify_family_objects(len(l_model_bb_res))

        l_family_result   = family_result.replace("```json", "").replace("```", "").strip()
        l_family_result = json.loads(l_family_result)

        l_model_family_classification = []
        l_model_family_description    = []
        l_model_bb                    = []

        for jj, family_result in enumerate(l_family_result['images']):

            family_classification = family_result['classification']
            family_description    = family_result['visual_evidence']
            bb                    = l_model_bb_res[jj]['box_2d']

            l_model_family_classification .append(family_classification)
            l_model_family_description    .append(family_description)
            l_model_bb                    .append(bb)

        modelResult.add(jpg_file, l_model_bb, l_model_family_classification, l_model_family_description)

        if i == 20:
            break

    modelResult.save_to_pickle(pkl_result_file)



