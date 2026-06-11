from tqdm                             import tqdm

from Code_Train_B.BoundingBoxMatcher import BoundingBoxMatcher
from Code_Train_B.vlm_model           import VlmModel
from PIL                              import Image, ImageDraw, ImageFont
import pandas as pd
import ast
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





def calc_statistics(boundingBoxMatcher, df):

    jpg_file       = df['jpg_file'].values[0]

    if jpg_file == '1_524400_186_14-51-38.jpg':
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

def print_statisics(df):
    from sklearn.metrics import confusion_matrix

    bb_count = df.bb_count.sum()
    gt_count = df.gt_count.sum()
    print(f"BB: {bb_count} | {gt_count}, {(bb_count/gt_count):.2f})%")
    l_gt_fam    = df.gt_family.values
    l_model_fam = df.model_family.values

    print(set(l_gt_fam))

    y_true = l_gt_fam
    y_pred = l_model_fam

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentages by row (each true class sums to 100%)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Pretty print
    labels = sorted(set(y_true) | set(y_pred))
    df = pd.DataFrame(cm_pct, index=labels, columns=labels)

    print(df.round(2))


def get_results(pkl_result_file):

    boundingBoxMatcher   = BoundingBoxMatcher(threshold=0.8, criterion='iog')
    l_statistics_results = []
    df                   = pd.read_pickle(pkl_result_file)

    for i in range(len(df)):
        df_current = df[df.jpg_file == df.jpg_file[i]]
        # draw_result(df_current, f'{jpg_files_path}/Images')
        # if df_current.jpg_file.values[0] == '2_924400_1136_09-56-59.jpg':
        #     print("")
        statistics_results = calc_statistics(boundingBoxMatcher, df_current)
        l_statistics_results.append(statistics_results)

    df_statistics_results = pd.concat(l_statistics_results)
    print_statisics(df_statistics_results)
    exit(0)

if __name__ == "__main__":

    RUN_MODEL   = False
    DB_TYPE     = "Validation" # "Train" / "Validation"
    USE_MOLMO   = True

    molmo_fname = ""
    if USE_MOLMO:
        molmo_fname = "_molmo"

    if DB_TYPE == "Train":
        jpg_files_path    = '/home/amitli/datasets/DOR_6/Train_B/Database'
        pkl_result_file   = f'/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/train_results{molmo_fname}.pkl'
        pkl_input_db_file = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/train_db.pkl'
    else:

        l_test_list = ['1_564400_419_14-50-09.jpg', '1_564400_137_14-50-09.jpg', '1_524400_420_14-51-3.jpg']


        jpg_files_path    = '/home/amitli/datasets/DOR_6/Train_B/validation'
        pkl_result_file   = f'/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_results{molmo_fname}.pkl'
        pkl_input_db_file = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_db.pkl'

    if RUN_MODEL is False:
       get_results(pkl_result_file)

    # df                      = pd.read_pickle(pkl_input_db_file)
    # df                      = df[df['num_gt'] > 0]
    # df                      = df.sample(n=50)
    # df.to_pickle(pkl_input_db_file.replace('.pkl', '_50_samples.pkl'))
    df                      = pd.read_pickle(pkl_input_db_file.replace('.pkl', '_50_samples.pkl'))
    modelResult             = ModelResult()

    vlmModel     = VlmModel(use_molmo_crop=USE_MOLMO)
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



