from transformers import AutoImageProcessor, AutoModel
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import umap
import plotly.express as px
import matplotlib.pyplot as plt

from app_config.settings import CSV_TRAIN_EMBEDDING_FILE, TRAIN_CROP_FILES


def load_model(model_name , device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    return processor, model


def create_embeddings(model, processor, device, csv_path, input_files_path, output_path):

    df         = pd.read_csv(csv_path)
    l_emb      = []
    l_jpg_file = []
    l_gt       = []


    for i in tqdm(range(len(df))):

        jpg_file  = os.path.basename(df['jpg_file'].values[i])
        if 'gt' in df.columns:
            gt = df['gt'].values[i]
        else:
            gt = "unknown"
        full_path = f"{input_files_path}/{jpg_file}"

        try:
            image = Image.open(full_path).convert("RGB")
        except FileNotFoundError:
            print(f"Cant load image: {full_path}")
            exit()

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings       = outputs.pooler_output
        embedding_vector = embeddings.reshape(-1).cpu().numpy()
        l_emb      .append(embedding_vector)
        l_jpg_file.append(jpg_file)
        l_gt      .append(gt)

    df_res = pd.DataFrame({"jpg_file": l_jpg_file, "gt": l_gt, "embedding": l_emb})
    df_res.to_csv(output_path, index=False)


def run_umap(embedding_file_name, output_file_name):
    df                 = pd.read_csv(embedding_file_name)
    df[f'embedding']   = df[f'embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    l_emb              = np.vstack(df[f'embedding'].values)

    reducer      = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    X_umap       = reducer.fit_transform(l_emb)
    df['umap_x'] = X_umap[:, 0]
    df['umap_y'] = X_umap[:, 1]

    df.to_csv(output_file_name, index=False)

def plot_umap_df(df_file_name):
    df        = pd.read_csv(df_file_name)
    color_map = {
        'SA-22'  : 'red',
        'SCUD'   : 'green',
        'T-90'   : 'blue',
    }

    # Map colors to df['gt']
    colors = df['gt'].map(color_map)

    plt.figure(figsize=(8, 6))
    plt.scatter(df['umap_x'], df['umap_y'], c=colors, s=40, alpha=0.8)

    plt.xlabel('UMAP X')
    plt.ylabel('UMAP Y')
    plt.title(f'UMAP projection colored by GT ({len(df)} Samples)')
    plt.grid(True)

    # Add legend
    for cls, color in color_map.items():
        plt.scatter([], [], c=color, label=cls)
    plt.legend(title='GT')

    plt.show()


def plot_umap_plotly_df(df_file_name, plot_file_name):
    df          = pd.read_csv(df_file_name)
    df['gt']    = df['gt'].fillna('Other')
    df['hover'] = df["jpg_file"]
    fig = px.scatter(df, x='umap_x', y='umap_y', color='gt', hover_name="hover", title=f"#Samples: {len(df)}")


    fig.write_html(plot_file_name)

    fig.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    processor, model = load_model(f"facebook/dinov2-base", device)
    create_embeddings(model, processor, device, "/home/amitli/repo/dor6_vision/Dataset/train.csv",
                      TRAIN_CROP_FILES,
                      CSV_TRAIN_EMBEDDING_FILE)

    run_umap(CSV_TRAIN_EMBEDDING_FILE,
             "/home/amitli/repo/dor6_vision/Dataset/umap_train_crop.csv")

    plot_umap_df("/home/amitli/repo/dor6_vision/Dataset/umap_train_crop.csv")
    plot_umap_plotly_df("/home/amitli/repo/dor6_vision/Dataset/umap_train_crop.csv",
                        "/home/amitli/repo/dor6_vision/Dataset/train_crop.html")