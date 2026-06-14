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
        with open("Prompts/image_description_prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read()

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

    def get_image_description(self, image_path, max_soft_tokens):

        prompt = self._prepare_image_description_prompt(image_path)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=prompt,
            temperature=0.0,
            top_p=1,
            seed=42,
            extra_body={
                "mm_processor_kwargs": {
                    "max_soft_tokens": max_soft_tokens
                }
            }
        )
        res_text = response.choices[0].message.content
        return res_text


def get_few_shots_files():
    FEW_SHOTS_FOLDER = '/home/amitli/repo/dor6_vision/Code_Train_B/Few_Shots/Targets/'
    LAUNCHER_SCUD_EXAMPE = rf'{FEW_SHOTS_FOLDER}/Scud/2_884400_119_09-57-12.jpg'
    LAUNCHER_ISKANDER_EXAMPLE = f'{FEW_SHOTS_FOLDER}/Iskander/3_1364400_384_09-59-20.jpg'
    LAUNCGER_SS_21_EXAMPLE = f'{FEW_SHOTS_FOLDER}/SS-21/3_1324400_366_09-58-46.jpg'  # ??? no missile

    ANTI_AIRCRAFT_SA_17_EXAMPLE = f'{FEW_SHOTS_FOLDER}/SA-17/2_884400_441_09-57-12.jpg'
    ANTI_AIRCRAFT_SA_22_EXAMPLE = f'{FEW_SHOTS_FOLDER}/SA-22/1_524400_500_09-56-15.jpg'
    ANTI_AIRCRAFT_TIN_SHIELD_EXAMPLE = f'{FEW_SHOTS_FOLDER}/Tin_Shield/3_1364400_274_09-59-20.jpg'
    ANTI_AIRCRAFT_GRAVE_STONE_EXAMPLE = f'{FEW_SHOTS_FOLDER}/Grave_Stone/3_1244400_166_09-58-27.jpg'
    ANTI_AIRCRAFT_BIG_BIRD_EXAMPLE = f'{FEW_SHOTS_FOLDER}/Big_Bird/3_1324400_343_09-58-46.jpg'

    TANK_T_90_EXAMPLE = f'{FEW_SHOTS_FOLDER}/T-90/3_1284400_190_09-59-02.jpg'

    l_few_shots_files = [LAUNCHER_SCUD_EXAMPE,
                         LAUNCHER_ISKANDER_EXAMPLE,
                         LAUNCGER_SS_21_EXAMPLE,
                         ANTI_AIRCRAFT_SA_17_EXAMPLE,
                         ANTI_AIRCRAFT_SA_22_EXAMPLE,
                         ANTI_AIRCRAFT_TIN_SHIELD_EXAMPLE,
                         ANTI_AIRCRAFT_GRAVE_STONE_EXAMPLE,
                         ANTI_AIRCRAFT_BIG_BIRD_EXAMPLE,
                         TANK_T_90_EXAMPLE]

    return l_few_shots_files

if __name__ == "__main__":
    imageDescription = ImageDescription()
    l_files = get_few_shots_files()
    for target_file in l_files:
        print(f'-'*50)
        print(f'{target_file}')
        result = imageDescription.get_image_description(target_file, 280)
        print(result)
        print("\n")

        # 1. **Visual Evidence:** Wheeled platform, no visible radar, carrying a single large missile.
        #
        # 2. **Visual Evidence:** No missiles or multiple missiles visible; radar or radar array may be present; vehicle appears configured for target tracking.
        #
        # 3. **Visual Evidence:** Continuous tracks (tracked vehicle), no visible radar, equipped with a large main gun barrel.




    # l_targets = glob.glob('/home/amitli/repo/dor6_vision/Code_Train_B/Few_Shots/Targets/*')
    # for target_folder in l_targets:
    #     l_target_files = glob.glob(target_folder + '/*.jpg')
    #     for target_file in l_target_files:
    #         result = imageDescription.get_image_description(target_file)
    #         print("\n")
    #         print("*" * 50)
    #         print(result)
    #         print("\n")
    #     break

