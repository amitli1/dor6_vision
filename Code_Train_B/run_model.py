from tqdm                             import tqdm
from Code_Train_B.BoundingBoxMatcher  import BoundingBoxMatcher
from Code_Train_B.vlm_model           import VlmModel
from PIL                              import Image, ImageDraw, ImageFont
from sklearn.metrics                  import confusion_matrix
import pandas as pd
import numpy  as np
import ast
import json

from app_config.settings import FONT_FILE


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





def calc_statistics(boundingBoxMatcher, df):

    jpg_file       = df['jpg_file'].values[0]

    if jpg_file == '1_564400_524_14-50-09.jpg':
        None

    gt_count       = df['gt_count'].values[0]
    l_gt_bb        = df['gt_bb'].values[0]
    l_gt_family    = df['gt_family'].values[0]
    l_model_family = df['model_family'].values[0]
    l_model_bb     = df['model_bb'].values[0]

    res_array_size     = max(len(l_gt_bb), len(l_model_bb))
    matching_results   = boundingBoxMatcher.match(l_model_bb, l_gt_bb)

    l_res_gt_family    = l_gt_family
    l_res_model_family = ['Not_Found'] * len(l_gt_family)

    bb_count          = 0
    l_tmp_gt_index    = []
    l_tmp_model_index = []
    l_tmp_model_fam   = []
    l_tmp_bb_found    = [0] * len(l_model_bb)
    for matcing in matching_results:
        match_gt_index = matcing["matched_gt_index"]
        if match_gt_index != -1:
            bb_count     = bb_count + 1
            model_index  = matcing["model_index"]
            model_family = l_model_family[model_index].replace('vehicle', '').strip()

            l_tmp_gt_index   .append(match_gt_index)
            l_tmp_model_index.append(model_index)
            l_tmp_model_fam  .append(model_family)

            l_tmp_bb_found[model_index] = 1

    # --- fill the founded gt wtih the model family
    for i in range(len(l_tmp_gt_index)):
        gt_index    = l_tmp_gt_index   [i]
        model_fam   = l_tmp_model_fam  [i]
        l_res_model_family[gt_index] = model_fam


    # fill the rest of the model
    for model_index in range(len(l_tmp_bb_found)):
        if l_tmp_bb_found[model_index] == 0:
            l_res_gt_family   .append("NOT_IN_GT")
            model_pred  = l_model_family[model_index]
            model_pred  = model_pred.rstrip()
            l_res_model_family.append(model_pred)

    l_res_bb_count    = [bb_count] + [0] * ( len(l_res_model_family)-1)
    l_res_gt_count    = [gt_count] + [0] * (len(l_res_model_family) - 1)
    l_jpg_res         = [jpg_file] * len(l_res_model_family)


    df_results = pd.DataFrame({
                              "jpg_file"        : l_jpg_res,
                              "bb_count"        : l_res_bb_count,
                              "gt_count"        : l_res_gt_count,
                              "gt_family"       : l_res_gt_family,
                              "model_family"    : l_res_model_family
    })
    return df_results


def print_statisics_not_in_gt(df):
    df          = df[df.gt_family == 'NOT_IN_GT']
    l_gt_fam    = df.gt_family.values
    l_model_fam = df.model_family.values

    print(f'l_gt_fam    = {set(l_gt_fam)}')
    print(f'l_model_fam = {set(l_model_fam)}')

    y_true = l_gt_fam
    y_pred = l_model_fam

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentages by row (each true class sums to 100%)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Pretty print
    labels = sorted(set(y_true) | set(y_pred))
    df = pd.DataFrame(cm_pct, index=labels, columns=labels)

    df.index.name = "GT (Actual)"
    df.columns.name = "Predicted"
    print(df.round(2))

def print_statisics(df):

    # calc BB statiscs
    bb_count = df.bb_count.sum()
    gt_count = df.gt_count.sum()
    print(f"BB: {bb_count} | {gt_count}, {(bb_count/gt_count):.2f})%")

    df_tmp = df[['jpg_file', 'gt_count', 'bb_count']]
    #print(df_tmp.head(50))

    # 1. fillter and calc statiscs only on founded bouding boxes
    df = df[df.model_family != 'Not_Found']

    # show confustion matrix only on GT family:
    df = df[df['gt_family'].isin(['Anti aircraft', 'Launchers', 'Tank'])]

    l_gt_fam = df.gt_family.values
    l_model_fam = df.model_family.values

    # unique_true = np.unique(l_gt_fam)
    # unique_pred = np.unique(l_model_fam)
    # print(f'unique_true = {unique_true}')
    # print(f'unique_pred = {unique_pred}')

    y_true = l_gt_fam
    y_pred = l_model_fam

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentages by row (each true class sums to 100%)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Pretty print
    labels = sorted(set(y_true) | set(y_pred))
    df = pd.DataFrame(cm_pct, index=labels, columns=labels)

    df.index.name = "GT (Actual)"
    df.columns.name = "Predicted"
    print(df.round(2))

    # row - GT
    # col - Model


def get_results(pkl_result_file, bb_threshold):

    boundingBoxMatcher   = BoundingBoxMatcher(threshold=bb_threshold, criterion='iog')
    l_statistics_results = []
    df                   = pd.read_pickle(pkl_result_file)

    for i in range(len(df)):
        df_current = df[df.jpg_file == df.jpg_file[i]]
        statistics_results = calc_statistics(boundingBoxMatcher, df_current)
        l_statistics_results.append(statistics_results)

    df_statistics_results = pd.concat(l_statistics_results)
    print_statisics(df_statistics_results)
    #print_statisics_not_in_gt(df_statistics_results)
    return df_statistics_results


def save_crop(image_path, bb, min_crop_size, crop_path):

    image         = Image.open(image_path).convert("RGB")
    width, height = image.size
    l_crop_ratio  = []

    ymin, xmin, ymax, xmax = bb
    left   = (xmin / 1000) * width
    top    = (ymin / 1000) * height
    right  = (xmax / 1000) * width
    bottom = (ymax / 1000) * height


    crop_width  = int(right - left)
    crop_height = int(bottom - top)
    crop_size   = (crop_width * crop_height) / (width * height)
    l_crop_ratio.append(crop_size)

    if crop_width < min_crop_size:
        delta = (min_crop_size - crop_width) / 2
        left -= delta
        right += delta

    if crop_height < min_crop_size:
        delta = (min_crop_size - crop_height) / 2
        top -= delta
        bottom += delta

    crop_image = image.crop((left, top, right, bottom))
    crop_image.save(crop_path, "JPEG")



def create_crops(df, vlmModel):
    bb_threshold = 0.4
    boundingBoxMatcher = BoundingBoxMatcher(threshold=bb_threshold, criterion='iog')

    l_crop_jpg_file = []
    l_crop_model_family = []

    for i in tqdm(range(len(df))):

        jpg_file       = df.jpg_file.values[i]
        l_gt_targets   = df.targets.values[i]
        l_gt_bb        = df.bb.values[i]
        full_jpg_file  = f'{jpg_files_path}/Images/{jpg_file}'
        l_model_bb_res = vlmModel.get_list_of_bounding_boxes(full_jpg_file)
        l_model_bb_res = vlmModel.convert_bb_molmo_to_gemma4(l_model_bb_res, radius=50)
        l_crop_ratio   = vlmModel.create_crop_files(full_jpg_file, l_model_bb_res, min_crop_size=128)

        l_model = []
        for jj in range(len(l_model_bb_res)):
            bb = l_model_bb_res[jj]['box_2d']
            l_model.append(bb)

        matching_results = boundingBoxMatcher.match(l_model, l_gt_bb)

        crop_num = 0
        for matcing in matching_results:
            match_gt_index = matcing["matched_gt_index"]
            if match_gt_index != -1:
                crop_num = crop_num + 1
                model_index = matcing["model_index"]
                l_crop = l_model[model_index]
                gt = l_gt_targets[match_gt_index]

                crop_path= "/home/amitli/repo/dor6_vision/Code_Train_B/validation_crops/"
                crop_file = jpg_file.replace('.jpg', f'_c_{crop_num}.jpg')
                full_crop = f"{crop_path}/{crop_file}"
                save_crop(full_jpg_file, l_crop, 128, full_crop)

                l_crop_jpg_file.append(full_crop)
                l_crop_model_family.append(gt)

    df_crops = pd.DataFrame({'crop_jpg_file': l_crop_jpg_file, 'gt_target': l_crop_model_family})
    df_crops.to_pickle(r'/home/amitli/repo/dor6_vision/Code_Train_B/validation_crops/crop.pkl')


if __name__ == "__main__":

    RUN_MODEL    = False
    DB_TYPE      = "Validation" # "Train" / "Validation"
    USE_MOLMO    = False
    DRAW_RESULTS = True
    CREATE_CROPS = False

    molmo_fname = ""
    if USE_MOLMO:
        molmo_fname = "_molmo"

    if DB_TYPE == "Train":
        jpg_files_path    = '/home/amitli/datasets/DOR_6/Train_B/Database'
        pkl_result_file   = f'/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/train_results{molmo_fname}.pkl'
        pkl_input_db_file = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/train_db.pkl'
    else:

        jpg_files_path    = '/home/amitli/datasets/DOR_6/Train_B/validation'
        pkl_result_file   = f'/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_results{molmo_fname}.pkl'
        pkl_input_db_file = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_db.pkl'

    if DRAW_RESULTS:
        file_name = '1_564400_524_14-50-09.jpg'
        file_name = '1_564400_141_14-50-09.jpg'
        file_name = '1_444400_462_14-49-59.jpg'
        df        = pd.read_pickle(pkl_result_file)
        df_sample = df[df.jpg_file == file_name]
        draw_result(df_sample, f"{jpg_files_path}/Images")
        exit(0)

    if (RUN_MODEL is False) and (CREATE_CROPS is False):
       get_results(pkl_result_file, bb_threshold=0.4)

    # df                      = pd.read_pickle(pkl_input_db_file)
    # df                      = df[df['num_gt'] > 0]
    # df                      = df.sample(n=100)
    # df.to_pickle(pkl_input_db_file.replace('.pkl', '_100_samples.pkl'))

    df                      = pd.read_pickle(pkl_input_db_file.replace('.pkl', '_100_samples.pkl'))
    modelResult             = ModelResult()
    vlmModel                = VlmModel(use_molmo_crop=USE_MOLMO)

    if CREATE_CROPS:
        create_crops(df, vlmModel)
        exit(0)

    for i in tqdm(range(len(df))):
        jpg_file       = df.jpg_file.values[i]
        l_gt_targets   = df.targets.values[i]
        l_gt_bb        = df.bb.values[i]
        full_jpg_file  = f'{jpg_files_path}/Images/{jpg_file}'
        l_model_bb_res = vlmModel.get_list_of_bounding_boxes(full_jpg_file)
        l_model_bb_res = vlmModel.convert_bb_molmo_to_gemma4(l_model_bb_res, radius=50)
        l_crop_ratio   = vlmModel.create_crop_files(full_jpg_file, l_model_bb_res, min_crop_size=128)
        family_result  = vlmModel.classify_family_objects(len(l_model_bb_res))


        if type(family_result) == str:
            l_family_result   = family_result.replace("```json", "").replace("```", "").strip()
            l_family_result = json.loads(l_family_result)
        else:
            l_family_result = family_result

        l_model_family_classification = []
        l_model_family_description    = []
        l_model_bb                    = []

        for jj, family_result in enumerate(l_family_result['images']):

            family_classification = family_result['classification']
            family_classification = family_classification.replace('vehicle', '')
            family_description    = family_result['visual_evidence']
            bb                    = l_model_bb_res[jj]['box_2d']

            l_model_family_classification .append(family_classification)
            l_model_family_description    .append(family_description)
            l_model_bb                    .append(bb)

        modelResult.add(jpg_file, l_model_bb, l_model_family_classification, l_model_family_description)


    modelResult.save_to_pickle(pkl_result_file)



