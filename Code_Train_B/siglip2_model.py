from PIL          import Image
from transformers import AutoModel, AutoProcessor, pipeline
import os
import torch

class Siglip2Model():

    def __init__(self):
        model_id              = "google/siglip2-giant-opt-patch16-384"
        self.TMP_FILES_FOLDER = '/home/amitli/repo/dor6_vision/Code_Train_B/TMP_FOLDER/'
        self.device           = "cuda" if torch.cuda.is_available() else "cpu"
        self.model            = AutoModel.from_pretrained(model_id).to(self.device)
        self.processor        = AutoProcessor.from_pretrained(model_id)

    def classify_family_objects(self, num_of_objects):

        l_obj_results   = []
        for i in range(num_of_objects):
            crop_file_path = f"{self.TMP_FILES_FOLDER}/crop_{i + 1}.jpg"
            image = Image.open(crop_file_path)

            LAUNCHERS     = "Launchers vehicle"
            ANTI_AIRCRAFT = "Anti aircraft vehicle"
            TANK          = "Tank"
            candidate_labels = [LAUNCHERS, ANTI_AIRCRAFT, TANK]
            # Formulating a descriptive prompt often helps alignment accuracy
            texts = [f"This is a simulation image of {label}." for label in candidate_labels]

            # Note: For SigLIP 2, padding='max_length' and max_length=64 are defaults used during pre-training
            inputs = self.processor(
                text=texts,
                images=image,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            # 4. Model forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

                # SigLIP outputs similarity scores in logits_per_image
                logits_per_image = outputs.logits_per_image

                # SigLIP was trained explicitly with a Sigmoid loss rather than Softmax!
                probs = torch.sigmoid(logits_per_image).squeeze()

            # 5. Map probabilities back to classes
            max_prob = 0
            max_label = ""
            for label, prob in zip(candidate_labels, probs):
                if prob >= max_prob:
                    max_prob  = prob
                    max_label = label

            json_object = {"classification": max_label.replace("vehicle", "").strip(),
                           "image_index": i,
                           "visual_evidence": ""}
            l_obj_results.append(json_object)

        l_obj_results = {"images": l_obj_results}
        return l_obj_results




