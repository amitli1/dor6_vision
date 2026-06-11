from PIL    import Image, ImageDraw, ImageFont
from openai import OpenAI

import base64
import json
import os
import re

class VlmModel():

    def __init__(self, use_molmo_crop=False):
        self.MODEL    = "gemma-4-31b-it"
        self.BASE_URL = "http://localhost:9000/v1"
        self.client   = OpenAI(
            api_key  = "gpustack_a853c8a4cf87ee4b_6d6d0ade6d71fbadf5b015e04fb5e825",
            base_url = "http://10.53.160.148/v1"
        )
        self.molmo_client     = OpenAI(api_key="EMPTY", base_url="http://localhost:9100/v1")
        self.TMP_FILES_FOLDER = '/home/amitli/repo/dor6_vision/Code_Train_B/TMP_FOLDER/'
        self.use_molmo_crop = use_molmo_crop


    def _encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _add_text_line(self, content, text):
        content.append({"type": "text", "text": f"{text}"})

    def _add_image_line(self, content, image_path):
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"}})

    def _get_all_bb(self, target_img):
        base_prompt = ("Return all the bounding boxes of the military vehicles in this military simulation image "
                       "and classify it as 'Launchers vehicle' or 'Anti aircraft vehicle' or 'Tank' or 'Other'")

        content = []
        self._add_text_line(content, base_prompt)
        self._add_image_line(content, target_img)

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        return messages

    def molmo_get_all_points_prompt(self, target_img):

        def encode_image(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Locate all the military vehicles in this image. Return only their locations"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(target_img)}"}}
                ],
            }
        ]
        return messages

    def extract_molmo_coords(self, text):

        # Regex searches for the content inside 'coords="..."'
        match = re.search(r'coords="([\d\s]+)"', text)

        if match:
            # Get the string of numbers
            coord_string = match.group(1)
            # Split by whitespace and convert to integers
            coords = [int(c) for c in coord_string.split()]
            return coords

        return []

    def molmo_get_list_of_points(self, full_image_path):
        messages = self.molmo_get_all_points_prompt(full_image_path)
        response = self.molmo_client.chat.completions.create(
            model="allenai/Molmo2-4B",
            messages=messages,
            temperature=0.0,
            top_p=1,
            seed=42,
        )
        res_text = response.choices[0].message.content
        res_text = self.extract_molmo_coords(res_text)
        # res_json = convert_point_to_boxes(res_text)

        return res_text

    def convert_bb_molmo_to_gemma4(self, l_model_bb_res, radius):
        if self.use_molmo_crop is False:
            return l_model_bb_res

        l_gemma_results = []

        for i in range(1, len(l_model_bb_res), 3):
            point_id = l_model_bb_res[i]
            x        = l_model_bb_res[i + 1]
            y        = l_model_bb_res[i + 2]

            l_gemma_results.append({
                "id": point_id,
                "box_2d": [
                    y - radius,  # ymin
                    x - radius,  # xmin
                    y + radius,  # ymax
                    x + radius,  # xmax
                ]
            })

        return l_gemma_results

    def get_list_of_bounding_boxes(self, full_image_path):

        if self.use_molmo_crop:
            return self.molmo_get_list_of_points(full_image_path)

        messages = self._get_all_bb(full_image_path)
        schema = {
            "type": "object",
            "properties": {
                "vehicles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "classification": {
                                "type": "string",
                                "enum": ["Launchers vehicle", "Anti aircraft vehicle", "Tank", "Other"]
                            },
                            "bounding_box": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Normalized coordinates [ymin, xmin, ymax, xmax]",
                                "minItems": 4,
                                "maxItems": 4
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["classification", "bounding_box"]
                    }
                }
            },
            "required": ["vehicles"]
        }
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=0.0,
            top_p=1,
            seed=42,
            extra_body={
                "guided_json": schema,
                "mm_processor_kwargs": {
                    "max_soft_tokens": 1120
                }
            }
        )
        res_text = response.choices[0].message.content
        res_text = res_text.replace("```json", "").replace("```", "").strip()
        try:
            res_json = json.loads(res_text)
        except Exception as e:
            res_json = {}
            print(f"Cant get bounding boxes, File: {os.path.basename(full_image_path)}, Got: {res_text}")
        return res_json


    def create_crop_files(self, image_path, model_bb_json_res, min_crop_size):

        image         = Image.open(image_path).convert("RGB")
        width, height = image.size
        l_crop_ratio  = []

        for i, bb in enumerate(model_bb_json_res):
            ymin, xmin, ymax, xmax = bb['box_2d']
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
            crop_image.save(f"{self.TMP_FILES_FOLDER}/crop_{i + 1}.jpg", "JPEG")

        return l_crop_ratio

    def _get_family_classification_prompt(self, num_of_objects):


        # LAUNCHERS = "Class_1"
        # ANTI_AIRCRAFT = "Class_2"
        # TANK = "Class_3"

        LAUNCHERS     = "Launchers vehicle"
        ANTI_AIRCRAFT = "Anti aircraft vehicle"
        TANK          = "Tank"

        content = []
        self._add_text_line(content, 'You are an expert in identifying military vehicles from simulation images')
        self._add_text_line(content,f"You receive {num_of_objects} patches of images from a military simulation and you must classify the military vehicle in the image, based only on the vehicle structure and mounted weapon system. Ignore background, terrain, and camera angle.")
        self._add_text_line(content,"CRITICAL RULE: Accuracy is more important than identification. If structural features are missing or ambiguous, you MUST label as 'Uncertain' or 'none'.")
        self._add_text_line(content,"Base your decision ONLY on visible structural features (e.g., wheels vs tracks, turret type, missile placement, number of missiles).")
        self._add_text_line(content, "Do NOT rely on color, background")

        self._add_text_line(content, "VISUAL DIFFERENTIATORS")

        self._add_text_line(content, f"{LAUNCHERS}")
        self._add_text_line(content, "1. Rocket / Missile Launchers / MLRS")
        self._add_text_line(content, "2. Deliver long-range missiles or rockets, usually indirect fire")
        self._add_text_line(content,"3. Long tube arrays mounted on a mobile vehicle")

        self._add_text_line(content, f"{ANTI_AIRCRAFT}")
        self._add_text_line(content, "1. A truck‑mounted system")
        self._add_text_line(content, "2. Anti-aircraft vehicles often have visible radar or sensor systems mounted on top.")

        self._add_text_line(content, f"{TANK}")
        self._add_text_line(content, "1. Main Battle Tanks")
        self._add_text_line(content, "2. No radar / missile tubes:")

                # --- no guess
        # add_text_line(content, 'If an image is too blurry to identify, label it as "Uncertain"')
        self._add_text_line(content, "If a clear object from the known classes is visible → output the class")
        self._add_text_line(content, 'If multiple classes are plausible or visibility is poor → output "Uncertain"')
        self._add_text_line(content, "Do not guess. If you are not sure with high certainty → output 'Uncertain'")

        self._add_text_line(content, f"TASK: Classify the following {num_of_objects} images. For each image:")
        self._add_text_line(content, "1. Describe what you see: 'I see [wheels/tracks], [missiles/gun], [radar/no radar]'.")
        self._add_text_line(content, "2. State if it meets all criteria or meets a REJECT condition.")
        self._add_text_line(content, f"3. Provide final classification: ['{LAUNCHERS}', '{ANTI_AIRCRAFT}', '{TANK}', 'none', 'Uncertain'].")

        # 5. RESPONSE FORMAT INSTRUCTIONS
        self._add_text_line(content, "RESPONSE FORMAT:")
        self._add_text_line(content, "You must return a JSON object containing an array of 'images'.")
        self._add_text_line(content,"For each image, you MUST first provide 'visual_evidence' describing the wheels/tracks and weapon systems before giving the 'classification'.")

        for i in range(num_of_objects):
            self._add_text_line(content, f"Image: {i + 1}")
            crop_file_path = f"{self.TMP_FILES_FOLDER}/crop_{i + 1}.jpg"
            self._add_image_line(content, crop_file_path)

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        return messages

    def classify_family_objects(self, num_of_objects):
        messages = self._get_family_classification_prompt(num_of_objects)

        schema = {
            "type": "object",
            "properties": {
                "images": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "visual_evidence": {
                                "type": "string",
                                "description": "List seen features: wheels/tracks, number/placement of missiles."
                            },
                            "classification": {
                                "type": "string",
                                "enum": ["Class_1", "Class_2", "Class_3", "none", "Uncertain"]
                            }
                        },
                        "required": ["visual_evidence", "classification"]
                    }
                }
            },
            "required": ["images"]
        }

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=0.0,
            top_p=1,
            seed=42,
            extra_body={
                "guided_json": schema,
                "mm_processor_kwargs": {
                    "max_soft_tokens": 280  # 70, 140, 280
                }
            }
        )
        res_text = response.choices[0].message.content
        return res_text



if __name__ == "__main__":

    #jpg_full_file = '/home/amitli/datasets/DOR_6/Train_B/validation/Images/1_564400_419_14-50-09.jpg'
    #jpg_full_file = '/home/amitli/datasets/DOR_6/Train_B/validation/Images/1_564400_137_14-50-09.jpg'
    jpg_full_file  = '/home/amitli/datasets/DOR_6/Train_B/validation/Images/1_524400_420_14-51-38.jpg'

    vlm_model      = VlmModel(use_molmo_crop=True)
    l_model_bb_res = vlm_model.get_list_of_bounding_boxes(jpg_full_file)
    l_model_bb_res = vlm_model.convert_bb_molmo_to_gemma4(l_model_bb_res, radius=50)
    l_crop_ratio   = vlm_model.create_crop_files(jpg_full_file, l_model_bb_res, min_crop_size=128)
    print(l_model_bb_res)
    print(l_crop_ratio)