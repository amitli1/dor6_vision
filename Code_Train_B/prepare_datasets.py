import glob
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import ast
from app_config.settings import FONT_FILE




def draw_image_with_gt(image_path, l_targets, l_bb):
    img          = Image.open(image_path)
    jpg_name     = os.path.basename(image_path)
    draw         = ImageDraw.Draw(img)
    img_w, img_h = img.size
    font         = ImageFont.truetype(FONT_FILE, size=32)

    draw.text(
        (1, 1),
        f'{jpg_name}',
        fill="yellow",
        font= ImageFont.truetype(FONT_FILE, size=32)
    )

    for i in range(len(l_targets)):
        label                      = l_targets[i]
        x_center, y_center, bw, bh = l_bb[i]

        x_center = float(x_center)
        y_center = float(y_center)
        bw       = float(bw)
        bh       = float(bh)

        # Convert to pixel coordinates
        x_center *= img_w
        y_center *= img_h
        bw       *= img_w
        bh       *= img_h

        x1 = x_center - bw / 2
        y1 = y_center - bh / 2
        x2 = x_center + bw / 2
        y2 = y_center + bh / 2

        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        draw.text((x1 + 50, y1), label, fill="red", font=font)

    img.show()

def draw_box(image_path, l_cords, output_jpg_file=None, l_prediction=None, show_img=False):
    # Load the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    font = ImageFont.truetype(FONT_FILE, size=32)

    # pred_str = ""
    # for i in range(len(l_prediction)):
    #     pred_str += l_prediction[i] + "\n"
    # draw.text((1, 1), pred_str, fill="red", font=font)

    if type(l_cords) == str:
        l_cords = ast.literal_eval(l_cords)

    for i, coords in enumerate(l_cords):
        ymin, xmin, ymax, xmax = coords

        # 2. Convert from normalized (0-1000) to actual pixel values
        left = xmin * width / 1000
        top = ymin * height / 1000
        right = xmax * width / 1000
        bottom = ymax * height / 1000

        # 3. Draw the rectangle
        # PIL expects [xmin, ymin, xmax, ymax]
        draw.rectangle([left, top, right, bottom], outline="red", width=1)
        if l_prediction is not None:
            draw.text((left + 50, top), l_prediction[i].replace('none', 'Other'), fill="red", font=font)
        # draw.text((left+50, top), f"{i+1}", fill="red", font=font)

    if output_jpg_file:
        img.save(output_jpg_file)
    if show_img:
        img.show()




DICT_TARGETS = {"0": "SA-22",
           "1": "Scud",
           "2": "T-90",
           "3": "SS-21",
           "4": "Iskander",
           "5": "SA-17",
           "6": "Tin Shield",
           "7": "Grad",
           "8": "Big Bird",
           "9": "Grave Stone"}

DICT_FAMILY = {"Launchers": [1, 4, 3, 7],
               "Anti aircraft": [5, 0, 8, 6, 9],
               "Tank": [2]}

def get_family(target_name):

    tgt_number = -1
    for target_number in DICT_TARGETS.keys():
        if DICT_TARGETS[target_number] == target_name:
            tgt_number = int(target_number)

    for key in DICT_FAMILY.keys():
        if tgt_number in DICT_FAMILY[key]:
            target_family = key

    return target_family


def convert_gt_number_to_gt_string(target_number):

    target_family = ""
    for key in DICT_FAMILY.keys():
        if target_number in DICT_FAMILY[key]:
            target_family = key

    if target_family == "":
        print("Error")

    return DICT_TARGETS[str(target_number)], target_family

def draw_histogram(df):
    # Histogram of targets
    import matplotlib.pyplot as plt

    # Targets
    target_counts = df['targets'].explode().value_counts()

    target_counts.plot(kind='bar', figsize=(10, 5))
    plt.title('Target Distribution')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # family
    family_counts = df['family'].explode().value_counts()

    family_counts.plot(kind='bar', figsize=(10, 5))
    plt.title('Family Distribution')
    plt.xlabel('Family')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # count
    family_counts = df['num_gt'].explode().value_counts()

    family_counts.plot(kind='bar', figsize=(10, 5))
    plt.title('Count Distribution')
    plt.xlabel('# of GT')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def prepare_data(dataset_path, pkl_path):

    dataset_path = f'{dataset_path}/Labels'
    l_txt_files  = glob.glob(f'{dataset_path}/*.txt')

    l_jpg_file = []
    l_targets  = []
    l_bb       = []
    l_num_gt   = []
    l_family   = []

    for txt_file in tqdm(l_txt_files):
        base_name = os.path.basename(txt_file)
        jpg_name  = base_name.replace('.txt', '.jpg')

        l_file_bb     = []
        l_file_gt     = []
        l_file_family = []

        with open(txt_file) as f:
            for line in f:
                parts = line.split()
                target, family = convert_gt_number_to_gt_string(int(parts[0]))
                l_file_gt.append(target)
                l_file_family .append(family)
                l_file_bb.append(parts[1:])

        l_jpg_file    .append(jpg_name)
        l_targets     .append(l_file_gt)
        l_family .append(l_file_family)
        l_bb          .append(l_file_bb)
        l_num_gt      .append(len(l_file_bb))

    df = pd.DataFrame({'jpg_file': l_jpg_file, 'targets': l_targets, "family": l_family, 'bb': l_bb, 'num_gt': l_num_gt})
    df.to_pickle(pkl_path)


def get_jpg_per_classes(df):

    l_different_labels = set(df.targets.explode().values)
    l_different_labels = [x for x in l_different_labels if pd.notna(x)]
    l_different_labels.sort()

    l_eda_jpg_file = []
    l_eda_targets  = []
    l_eda_bb       = []
    l_eda_family   = []

    for target in tqdm(l_different_labels):
        l_jpg_names = df[df['targets'].apply(lambda x: target in x)]['jpg_file']
        for jpg_file in l_jpg_names:
            l_current_bb  = df[df.jpg_file == jpg_file]['bb'].values[0]
            l_current_tgs = df[df.jpg_file == jpg_file]['targets'].values[0]

            for i in range(len(l_current_tgs)):
                if l_current_tgs[i] == target:
                    l_eda_targets  .append(target)
                    l_eda_jpg_file .append(jpg_file)
                    l_eda_bb       .append(l_current_bb[i])
                    l_eda_family   .append(get_family(target))


    df_eda = pd.DataFrame({'jpg_file': l_eda_jpg_file,
                           'targets' : l_eda_targets,
                           'family'  : l_eda_family,
                           'bb'      : l_eda_bb})

    df_eda.to_pickle('train_b_eda.pkl')

    return df_eda





if __name__ == "__main__":


    train_dataset_path      = '/home/amitli/datasets/DOR_6/Train_B/Database'
    validation_dataset_path = '/home/amitli/datasets/DOR_6/Train_B/validation'


    prepare_data(train_dataset_path,      '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/train_db.pkl')
    prepare_data(validation_dataset_path, '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_db.pkl')

    df = pd.read_pickle('/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_db.pkl')
    draw_histogram(df)

