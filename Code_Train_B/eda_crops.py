from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import shutil

from Code_Train_B.prepare_datasets import get_family
from Code_Train_B.vlm_model import VlmModel
import json
import os

def plot_histogram(df):
    df["gt_target"].hist(bins=30, figsize=(8, 5))

    plt.xlabel("gt_target")
    plt.ylabel("Count")
    plt.title("Distribution of gt_target")
    plt.show()

def print_cm(df, precentage_flag):

    y_true = df.gt_family.values
    y_pred = df.model_family.values

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentages by row (each true class sums to 100%)
    if precentage_flag:
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        cm = cm_pct

    # Pretty print
    labels = sorted(set(y_true) | set(y_pred))
    df = pd.DataFrame(cm, index=labels, columns=labels)

    df.index.name = "GT (Actual)"
    df.columns.name = "Predicted"
    print(df.round(2))


def predict_family_from_crops(df, pkl_results):
    vlmModel = VlmModel(use_molmo_crop=False)

    l_gt_target = []
    l_gt_family = []
    l_model_family = []
    l_crop_file = []
    l_family_desc = []

    for i in tqdm(range(len(df))):
        gt_target = df.gt_target.values[i]
        gt_family = get_family(gt_target)

        crop_jpg_file = df.crop_jpg_file.values[i]
        dest_file = "/home/amitli/repo/dor6_vision/Code_Train_B/TMP_FOLDER/crop_1.jpg"
        shutil.copy2(crop_jpg_file, dest_file)
        family_result = vlmModel.classify_family_objects(1, use_few_shots=True)
        if type(family_result) == str:
            l_family_result = family_result.replace("```json", "").replace("```", "").strip()
            l_family_result = json.loads(l_family_result)
        else:
            l_family_result = family_result
        family_classification = l_family_result['images'][0]['classification']
        family_classification = family_classification.replace('vehicle', '')
        family_classification = family_classification.strip()
        visual_evidence = l_family_result['images'][0]['visual_evidence']

        l_gt_target.append(gt_target)
        l_gt_family.append(gt_family)
        l_crop_file.append(crop_jpg_file)
        l_model_family.append(family_classification)
        l_family_desc.append(visual_evidence)

    df_results = pd.DataFrame({'crop_file': l_crop_file,
                               'gt_target': l_gt_target,
                               'gt_family': l_gt_family,
                               'model_family': l_model_family,
                               'model_family_desc': l_family_desc})

    df_results.to_pickle(pkl_results)


def plot_wrong(df_eda):
    l_crop_file = df_eda.crop_file.values
    l_gt = df_eda.gt_target.values

    l_names = [
        f"{os.path.basename(img_path)}_{gt}"
        for img_path, gt in zip(l_crop_file, l_gt)
    ]

    fig, axes = plt.subplots(5, 8, figsize=(16, 10))

    for ax, img_path, title in zip(axes.flat, l_crop_file, l_names):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for ax in axes.flat[len(l_crop_file):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    VALIDATION_CROP_FOLDER = '/home/amitli/repo/dor6_vision/Code_Train_B/validation_crops/'
    df = pd.read_pickle(rf'{VALIDATION_CROP_FOLDER}crop.pkl')
    #plot_histogram(df)
    pkl_results = '/home/amitli/repo/dor6_vision/Code_Train_B/Pickles/validation_crops_results_2.pkl'
    predict_family_from_crops(df, pkl_results)
    df_results = pd.read_pickle(pkl_results)
    print_cm(df_results, precentage_flag=False)
    print_cm(df_results, precentage_flag=True)

    # Predicted      Anti aircraft  Launchers  Tank  Uncertain  none
    # GT (Actual)
    # Anti aircraft             88         32     5          2     2
    # Launchers                  1         40     0          2     2
    # Tank                       2          1    42          0     0
    # Uncertain                  0          0     0          0     0
    # none                       0          0     0          0     0
    # Predicted      Anti aircraft  Launchers   Tank  Uncertain  none
    # GT (Actual)
    # Anti aircraft          68.22      24.81   3.88       1.55  1.55
    # Launchers               2.22      88.89   0.00       4.44  4.44
    # Tank                    4.44       2.22  93.33       0.00  0.00
    # Uncertain                NaN        NaN    NaN        NaN   NaN
    # none                     NaN        NaN    NaN        NaN   NaN


    # new few fhots:
    # Predicted      Anti aircraft  Launchers  Tank  Uncertain  none
    # GT (Actual)
    # Anti aircraft             17         57    10         42     3
    # Launchers                  0         40     0          3     2
    # Tank                       0          2    43          0     0
    # Uncertain                  0          0     0          0     0
    # none                       0          0     0          0     0
    # Predicted      Anti aircraft  Launchers   Tank  Uncertain  none
    # GT (Actual)
    # Anti aircraft          13.18      44.19   7.75      32.56  2.33
    # Launchers               0.00      88.89   0.00       6.67  4.44
    # Tank                    0.00       4.44  95.56       0.00  0.00
    # Uncertain                NaN        NaN    NaN        NaN   NaN
    # none                     NaN        NaN    NaN        NaN   NaN


    # 32 samples (gt = 'Anti aircraft', pred = 'Launchers'
    df_eda = df_results
    df_eda = df_eda[df_eda.gt_family == 'Anti aircraft']
    df_eda = df_eda[df_eda.model_family == 'Launchers']


    # for i in range(len(df_eda)):
    #     file = os.path.basename(df_eda.crop_file.values[i])
    #     target = df_eda.gt_target .values[i]
    #     desc = df_eda.model_family_desc.values[i]
    #     print(f'{file}: {target}, {desc}')
    #
    # plot_wrong(df_eda)








