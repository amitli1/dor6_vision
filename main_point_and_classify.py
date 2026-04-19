from transformers import AutoProcessor, AutoModelForImageTextToText
from sklearn.metrics import confusion_matrix, classification_report
import torch
from PIL import Image
import re
from tqdm import tqdm
import json
import pandas as pd

TMP_IMG_FILE = "/home/amitli/repo/dor6_vision/Dataset/tmp/tmp.jpg"

def load_molmo(model_id):
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
        use_fast=False
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

def create_pointing_prompt(target_image_path):
    image = Image.open(target_image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=f"point to the weapon system"),
                dict(type="image", image=image),
            ],
        }
    ]
    return messages

def create_classifcation_prompt(target_image_path):
    T_90_CROP_IMG_1_PATH  = "Dataset/few_shots/T-90/T-90_CROP_1.jpg"
    T_90_CROP_IMG_2_PATH  = "Dataset/few_shots/T-90/T-90_CROP_2.jpg"
    SA_22_CROP_IMG_1_PATH = "Dataset/few_shots/SA-22/SA-22_CROP_1.jpg"
    SA_22_CROP_IMG_2_PATH = "Dataset/few_shots/SA-22/SA-22_CROP_2.jpg"
    SCUD_CROP_IMG_1_PATH  = "Dataset/few_shots/SCUD/SCUD_CROP_1.jpg"
    SCUD_CROP_IMG_2_PATH  = "Dataset/few_shots/SCUD/SCUD_CROP_2.jpg"

    t_90_crop_img1  = Image.open(T_90_CROP_IMG_1_PATH).convert("RGB")
    t_90_crop_img2  = Image.open(T_90_CROP_IMG_2_PATH).convert("RGB")
    sa_22_crop_img1 = Image.open(SA_22_CROP_IMG_1_PATH).convert("RGB")
    sa_22_crop_img2 = Image.open(SA_22_CROP_IMG_2_PATH).convert("RGB")
    scud_crop_img1  = Image.open(SCUD_CROP_IMG_1_PATH).convert("RGB")
    scud_crop_img2  = Image.open(SCUD_CROP_IMG_2_PATH).convert("RGB")

    # T_90_IMG_1_PATH = "Dataset/few_shots/T-90/T-90_IMG_1.jpg"
    # T_90_IMG_2_PATH = "Dataset/few_shots/T-90/T-90_IMG_2.jpg"
    # SA_22_IMG_1_PATH = "Dataset/few_shots/SA-22/SA-22_IMG_1.jpg"
    # SA_22_IMG_2_PATH = "Dataset/few_shots/SA-22/SA-22_IMG_2.jpg"
    # SCUD_IMG_1_PATH = "Dataset/few_shots/SCUD/SCUD_IMG_1.jpg"
    # SCUD_IMG_2_PATH = "Dataset/few_shots/SCUD/SCUD_IMG_2.jpg"
    #
    # t_90_crop_img1 = Image.open(T_90_IMG_1_PATH).convert("RGB")
    # t_90_crop_img2 = Image.open(T_90_IMG_2_PATH).convert("RGB")
    # sa_22_crop_img1 = Image.open(SA_22_IMG_1_PATH).convert("RGB")
    # sa_22_crop_img2 = Image.open(SA_22_IMG_2_PATH).convert("RGB")
    # scud_crop_img1 = Image.open(SCUD_IMG_1_PATH).convert("RGB")
    # scud_crop_img2 = Image.open(SCUD_IMG_2_PATH).convert("RGB")

    target_image    = Image.open(target_image_path).convert("RGB")

    #
    #   class_0 : SA_22
    #   class_1: SCUD
    #   class_2: T-90

    messages = [
        {
            "role": "user",
            "content": [

                {"type": "image", "image": sa_22_crop_img1},
                {"type": "image", "image": sa_22_crop_img2},
                {"type": "image", "image": scud_crop_img1},
                {"type": "image", "image": scud_crop_img2},
                {"type": "image", "image": t_90_crop_img1},
                {"type": "image", "image": t_90_crop_img2},
                {"type": "image", "image": target_image},
                {
                    "type": "text",
                    "text": (
                        "Find the object in the image and classify it as either class_0, class_1, or class_2"
                        "I am providing 6 reference images. "
                        "In Image 1, the weapon system is a class_0. "
                        "In Image 2, the weapon system is a class_0. "
                        "In Image 3, the weapon system is a class_1. "
                        "In Image 4, the weapon system is a class_1. "
                        "In Image 5, the weapon system is a class_2. "
                        "In Image 6, the weapon system is a class_2. " 
                        "Now look at Image 7. Point to the weapon system and classify it as a class_0, class_1, or class_2."
                        "Return the result in a JSON-like format with the keys 'class'"
                    )
                }
            ]
        }
    ]

    return messages


def run_molmo_prediction(processor, model, prompt):
    inputs = processor.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_tokens      = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text        = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def get_pixel_coords(molmo_output, img_width, img_height):
    # Find all x and y patterns in the Molmo output string
    match = re.search(r'coords="([\d\s]+)"', molmo_output)
    if match:
        coords = list(map(int, match.group(1).split()))
        x_norm, y_norm = coords[-2], coords[-1]  # last two values
        x = (x_norm / 1000) * img_width
        y = (y_norm / 1000) * img_height
        return x_norm, y_norm, int(x), int(y)

    return None, None, None, None

def create_crop_image(molmo_prediction, image_path):
    image                 = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    _, _, x, y            = get_pixel_coords(molmo_prediction, img_width, img_height)
    if x is None:
        # no object
        return False

    crop_size  = 200
    left       = max(0, x - crop_size / 2)
    top        = max(0, y - crop_size / 2)
    right      = min(img_width, x + crop_size / 2)
    bottom     = min(img_height, y + crop_size / 2)

    crop_image = image.crop((left, top, right, bottom))
    # crop_image.show()
    crop_image.save(TMP_IMG_FILE, "JPEG")
    return True


def get_test_set(use_train=False):
    if use_train:
        df = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/train.csv')
        df_sa_22 = df[df['gt'] == 'SA-22']
        df_scud = df[df['gt'] == 'SCUD']
        df_t_90 = df[df['gt'] == 'T-90']

        df_sa_22 = df_sa_22.sample(n=100, replace=False)
        df_scud = df_scud.sample(n=100, replace=False)
        df_t_90 = df_t_90.sample(n=100, replace=False)

        l_files = list(df_sa_22.jpg_file.values) + list(df_scud.jpg_file.values) + list(df_t_90.jpg_file.values)
        l_gt    = list(df_sa_22['gt'].values)    + list(df_scud['gt'].values)    + list(df_t_90['gt'].values)
    else:
        test_path = "/home/amitli/repo/dor6_vision/Dataset/test_set/"
        files = ["frame_0.jpg", "frame_1530.jpg", "frame_3030.jpg", "frame_6030.jpg", "frame_7530.jpg",
                 "frame_9660.jpg", "frame_10350.jpg", "frame_14820.jpg", "frame_17850.jpg", "frame_14550.jpg",
                 "frame_12060.jpg", "frame_10260.jpg", "frame_6180.jpg", "frame_1080.jpg", "frame_12150.jpg",
                 "frame_12450.jpg", "frame_13350.jpg", "frame_13830.jpg"]
        l_gt = ["T-90", "T-90", "SCUD", "SA-22", "T-90", "SA-22", "SCUD", "SCUD", "T-90", "SCUD", "SCUD", "SA-22",
                "SA-22", "T-90", "SA-22", "SA-22", "SA-22", "SA-22"]

        l_files = []
        for file in files:
            l_files.append(f"{test_path}{file}")

    return l_files, l_gt


if __name__ == "__main__":


    processor,model  = load_molmo("allenai/Molmo2-4B")
    l_files, l_gt    = get_test_set(use_train=True)

    l_pred = []
    for i in tqdm(range(len(l_files))):
        full_image_path  = l_files[i]
        point_prompt     = create_pointing_prompt(full_image_path)
        molmo_pred       = run_molmo_prediction(processor, model, point_prompt)
        crop_ok          = create_crop_image(molmo_pred, full_image_path)
        if crop_ok is False:
            print(f"{full_image_path}, gt: {l_gt[i]} prompt: {point_prompt}, cant crop")
            pred_weapon = "Other"
        else:
            classify_prompt  = create_classifcation_prompt(TMP_IMG_FILE)
            molmo_pred_class = run_molmo_prediction(processor, model, classify_prompt)
            res_json        = json.loads(molmo_pred_class)
            pred_weapon     = res_json["class"]
            if pred_weapon == "class_0":
                pred_weapon = "SA-22"
            elif pred_weapon == "class_1":
                pred_weapon = "SCUD"
            elif pred_weapon == "class_2":
                pred_weapon = "T-90"
            else:
                pred_weapon = "Other"
        l_pred.append(pred_weapon)


    df_dor_results = pd.DataFrame({"jpg_file": l_files, "gt": l_gt, "class": l_pred})
    df_dor_results.to_csv("dor_results.csv", index=False)

    classes = ["SA-22", "SCUD", "T-90"]
    cm      = confusion_matrix(l_gt, l_pred, labels=classes)
    cm_df   = pd.DataFrame(cm, index=classes, columns=classes)
    print("Confusion Matrix:")
    print(cm_df)

