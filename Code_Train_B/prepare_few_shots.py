from PIL    import Image, ImageDraw, ImageFont
from openai import OpenAI
import glob
import base64
import json

from Testers.Create_crops import plot_few_shots


class ImageDescription:

    def __init__(self):
        self.MODEL = "gemma-4-31b-it"
        self.BASE_URL = "http://localhost:9000/v1"
        self.client = OpenAI(
            api_key="gpustack_a853c8a4cf87ee4b_6d6d0ade6d71fbadf5b015e04fb5e825",
            base_url="http://10.53.160.148/v1"
        )

    def _encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _add_text_line(self, content, text):
        content.append({"type": "text", "text": f"{text}"})

    def _add_image_line(self, content, image_path):
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"}})

    def _prepare_image_description_prompt(self, image_path):

        content = []
        prompt = """
            You are analyzing a single simulation image of a military vehicle or weapon system.
            
            Your task is to describe only what is visually observable in the image. Do NOT identify, classify, name, or guess the type, model, manufacturer, country, or intended role of the system.
            
            General rules:
            
            * Base your answer only on visible evidence in the image.
            * If a detail cannot be determined confidently, write "Not visible" or "Cannot determine from image".
            * Do not infer hidden components.
            * Do not speculate.
            * Do not use yes/no answers.
            * Do not assume an object exists unless visual evidence supports it.
            * Be precise and quantitative whenever possible.
            
            Provide your answer in the following format:
            
            ## Radar and Sensor Structures
            
            Describe every radar-like, antenna-like, dish-like, panel-like, or sensor structure that is visible.
            
            For each structure provide:
            
            * Approximate location on the vehicle (front, rear, left side, right side, roof, turret, mast, etc.)
            * Approximate size relative to the vehicle (small, medium, large, estimated percentage of vehicle height/width if possible)
            * Shape and appearance
            
            If no such structure is visible, state:
            "Radar or sensor structures not visibly identifiable."
            
            ## Missile Inventory
            
                Describe all missile-like objects that are visible.
            
            Provide:
            
                * Number of visible missiles
                * Number of launch tubes or launch containers if visible
                * Arrangement (single row, double row, clustered, stacked, etc.)
                * Approximate location on the vehicle                
            
            If the count is uncertain, explain why.
            
            ## Wheel Analysis
            
                Describe the wheel configuration.
            
            Provide:
            
                * Number of visible wheels
                * Estimated total wheel count for the vehicle (only if supported by visual evidence)
                * Wheel arrangement by side if visible
            
            If the total number cannot be determined, explain what is visible and what is occluded.
            
            ## Other Notable Visual Properties
            
                List any visually interesting or distinctive characteristics, such as:
            
                    * Large antennas
                    * Masts
                    * Stabilizers
                    * Launch platforms
                    * Turrets
                    * Camouflage patterns
                    * Support structures
                    * Unusual geometry
                    * Elevated components
                    * Containers or pods
                    * Tracks instead of wheels
                    * Any other prominent visual feature
            
                For each observation:
            
                    * Describe the feature
                    * Describe its location
                    * Explain the visual evidence supporting the observation
            
            ## Final Objective Description
            
                Provide a concise neutral description (3-8 sentences) summarizing the visible structure, layout, and major components of the system.
                
                Do not classify the system, identify its type, or infer its purpose.

        """
        self._add_text_line(content, prompt)
        self._add_text_line(content, f"Image to describe:")
        self._add_image_line(content, image_path)
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        return messages

    def get_image_description(self, image_path):

        prompt = self._prepare_image_description_prompt(image_path)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=prompt,
            temperature=0.0,
            top_p=1,
            seed=42,
            extra_body={
                "mm_processor_kwargs": {
                    "max_soft_tokens": 1120
                }
            }
        )
        res_text = response.choices[0].message.content
        return res_text

if __name__ == "__main__":
    imageDescription = ImageDescription()
    l_targets = glob.glob('/home/amitli/repo/dor6_vision/Code_Train_B/Few_Shots/Targets/*')
    for target_folder in l_targets:
        l_target_files = glob.glob(target_folder + '/*.jpg')
        for target_file in l_target_files:
            result = imageDescription.get_image_description(target_file)
            print("\n")
            print("*" * 50)
            print(result)
            print("\n")
        break

