from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import re
from app_config.settings import POINTING_IMAGE_WARMUP


class PointingAgent():

    def __init__(self):
        self.processor,self.model = self.load_molmo("allenai/Molmo2-4B")
        self.prompt = "point to the military vehicle"
        self.molmo_warmup()

    def load_molmo(self, model_id):
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

    def get_pixel_coords(self, molmo_output, img_width, img_height):
        # Find all x and y patterns in the Molmo output string
        match = re.search(r'coords="([\d\s]+)"', molmo_output)
        if match:
            coords = list(map(int, match.group(1).split()))
            x_norm, y_norm = coords[-2], coords[-1]  # last two values
            x = (x_norm / 1000) * img_width
            y = (y_norm / 1000) * img_height
            return x_norm, y_norm, int(x), int(y)

        return None, None, None, None

    def molmo_warmup(self):
        image_path = POINTING_IMAGE_WARMUP


        for i in range(2):
            generated_text, x, y = self.run_molmo_prediction(image_path)

    def run_molmo_prediction(self, image_path):

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="text", text=f"{self.prompt}"),
                    dict(type="image", image=image),
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048)

        generated_tokens      = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text        = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        img_width, img_height = image.size
        _, _, x, y = self.get_pixel_coords(generated_text, img_width, img_height)

        return generated_text, x, y
