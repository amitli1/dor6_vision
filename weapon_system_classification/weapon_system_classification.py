import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time

from app_config.settings import CSV_EMBEDDING_FILE


class WeaponSystemClassification:

    def __init__(self, use_knn):
        self.df_emb                = pd.read_csv(CSV_EMBEDDING_FILE)
        self.processor, self.model = self.load_dino_model('facebook/dinov2-base')
        self.df_emb.embedding      = self.df_emb.embedding.apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        self.l_embeddings          = np.vstack(self.df_emb.embedding.values)
        self.l_labels              = self.df_emb['gt'].values
        self.use_knn               = use_knn
        self.knn                   = NearestNeighbors(n_neighbors=30, metric='cosine', algorithm='brute')
        self.knn.fit(self.l_embeddings)



    def load_dino_model(self, model_id):
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model     = AutoModel.from_pretrained(model_id).to(device)
        return processor, model

    def get_embedding(self, image_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image  = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get the last hidden state and mean pool or take [CLS] token
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding

    def get_per_class_confidence(self, new_image_path):

        if self.use_knn is False:
            # 1. Get embedding for the new image
            new_emb  = self.get_embedding(new_image_path).reshape(1, -1)
            all_sims = cosine_similarity(new_emb, self.l_embeddings)[0]

            # 3. Group similarities by class and calculate the score
            class_scores = {}
            for label in ["SA-22", "SCUD", "T-90"]:
                # Find indices where the ground truth matches the label
                label_indices = np.where(self.l_labels == label)[0]

                # Get all similarity scores for this specific class
                specific_class_sims = all_sims[label_indices]

                # We take the top 5 highest similarities for this class and average them
                # This is more robust than taking the single best or the overall average
                top_scores          = np.sort(specific_class_sims)[-5:]
                class_scores[label] = np.mean(top_scores)
        else:
            new_emb  = self.get_embedding(new_image_path).reshape(1, -1)
            distances, indices = self.knn.kneighbors(new_emb)
            # 3. Process results
            class_scores = {label: [] for label in ["SA-22", "SCUD", "T-90"]}

            similarities = 1 - distances[0]
            neighbor_labels = self.l_labels[indices[0]]

            for sim, label in zip(similarities, neighbor_labels):
                if label in class_scores:
                    class_scores[label].append(sim)

            # Average the similarities for each class found in the top-K
            class_scores =  {
                label: (np.mean(scores) if scores else 0.0)
                for label, scores in class_scores.items()
            }

        return class_scores