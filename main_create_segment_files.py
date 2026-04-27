from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL                       import Image, ImageDraw
from tqdm                      import tqdm
from app_config.settings       import TRAIN_CROP_FILES, TRAIN_SEGMENT_FILES
import numpy                   as np
import                         os
import                         glob
import                         random



def plot_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img.show()

def get_mask_from_point(predictor, test_img, x, y):
    img = Image.open(test_img).convert("RGB")
    predictor.set_image(img)

    # Use the points from Step 1 as a prompt
    # input_labels=1 tells SAM these points represent the object (foreground)

    pixel_coords = []
    pixel_coords.append([x, y])
    pixel_coords = np.array(pixel_coords)
    masks, scores, logits = predictor.predict(
        point_coords=pixel_coords,
        point_labels=np.ones(len(pixel_coords)),
        multimask_output=False
    )
    mask = masks[0]  # The resulting binary mask (True for object, False for background)
    return mask


def run_sam2(sam_2_predictor, test_img, x, y):
    img       = Image.open(test_img).convert("RGB")
    sam_2_predictor.set_image(img)

    # Use the points from Step 1 as a prompt
    # input_labels=1 tells SAM these points represent the object (foreground)

    pixel_coords = []
    pixel_coords.append([x, y])
    pixel_coords = np.array(pixel_coords)
    masks, scores, logits = sam_2_predictor.predict(
        point_coords=pixel_coords,
        point_labels=np.ones(len(pixel_coords)),
        multimask_output=False
    )
    mask = masks[0]  # The resulting binary mask (True for object, False for background)
    return mask

def crop_and_background_removal(img_path, mask):
    mask = mask.astype(bool)
    img = Image.open(img_path).convert("RGB")

    # Create background-removed image
    image_np = np.array(img)
    image_np[~mask] = 0  # Set background pixels to black

    # Get Bounding Box for tighter cropping
    y_indices, x_indices = np.where(mask)
    bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]

    # Crop the background-removed image to the object
    isolated_object_img = Image.fromarray(image_np).crop(bbox)
    return isolated_object_img

def get_middle_point(img_path):
    image = Image.open(img_path).convert("RGB")
    img_width, img_height = image.size
    x = int(img_width/2)
    y = int(img_height/2)
    return x, y

if __name__ == '__main__':
    sam_2_predictor  = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
    l_crop_files     = glob.glob(TRAIN_CROP_FILES + "/*.jpg")
    random.shuffle(l_crop_files)

    for i, crop_file in enumerate(tqdm(l_crop_files)):
        x, y        = get_middle_point(crop_file)
        mask        = run_sam2(sam_2_predictor, crop_file, x, y)
        mask_img    = crop_and_background_removal(crop_file, mask)
        base_name   = os.path.basename(crop_file)
        output_path = f"{TRAIN_SEGMENT_FILES}/{base_name}"
        mask_img.save(output_path, format="JPEG")




