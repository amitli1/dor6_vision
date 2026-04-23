import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import time

from app_config.settings import CSV_TRAIN_EMBEDDING_FILE


class WeaponSystemClassification:

    def __init__(self, use_knn):
        self.df_emb                = pd.read_csv(CSV_TRAIN_EMBEDDING_FILE)
        self.processor, self.model = self.load_dino_model('facebook/dinov2-base')
        l_embeddings               = self.df_emb.embedding.apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        self.l_embeddings          = np.vstack(l_embeddings)
        self.l_labels              = self.df_emb['gt'].values
        #self.use_knn               = use_knn
        #self.knn                   = NearestNeighbors(n_neighbors=30, metric='cosine', algorithm='brute')
        #self.knn.fit(self.l_embeddings)



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

        new_emb         = self.get_embedding(new_image_path).reshape(1, -1)
        distances       = cdist(new_emb, self.l_embeddings, metric='cosine').flatten()
        K               = 10
        closest_indices = np.argsort(distances)[:K]
        neighbor_labels = self.l_labels[closest_indices]
        unique, counts  = np.unique(neighbor_labels, return_counts=True)
        relative_counts = counts / K
        class_scores    = dict(zip(unique, relative_counts))
        return class_scores