import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class EdaDrawTargets:
    def __init__(self, margin, dataset_path, df_eda):
        self.MARGIN       = margin
        self.dataset_path = dataset_path
        self.df_eda       = self._prepare_df(df_eda)

    def _prepare_df(self, df_eda):
        l_jpg    = []
        l_bb     = []
        l_target = []
        l_family = []

        for i in range(len(df_eda)):
            jpg_file = df_eda["jpg_file"].values[i]
            num_of_gt = df_eda["num_gt"].values[i]
            for jj in range(num_of_gt):
                bb       = df_eda["bb"].values[i][jj]
                target   = df_eda["targets"].values[i][jj]
                family   = df_eda["family"].values[i][jj]

                l_jpg.append(jpg_file)
                l_bb.append(bb)
                l_target.append(target)
                l_family.append(family)

        df = pd.DataFrame({"jpg_file": l_jpg,
                           "bb"      : l_bb,
                           "targets" : l_target,
                           "family"  : l_family})
        return df

    def _yolo_to_xyxy(self, bb, img_w, img_h):
        """
        bb: [x_center, y_center, w, h] normalized YOLO format
        returns: (x1, y1, x2, y2) in pixel coords
        """
        x_c, y_c, w, h = bb
        x_c = float(x_c)
        y_c = float(y_c)
        w = float(w)
        h = float(h)

        x_c *= img_w
        y_c *= img_h
        w *= img_w
        h *= img_h

        x1 = int(x_c - w / 2)
        y1 = int(y_c - h / 2)
        x2 = int(x_c + w / 2)
        y2 = int(y_c + h / 2)

        return x1, y1, x2, y2


    def _crop_with_margin(self, img, box, margin=50):
        x1, y1, x2, y2 = box
        w, h = img.size

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        return img.crop((x1, y1, x2, y2))


    def _get_targets_sub_sample(self, df_eda, num_of_samples):
        df_one_per_target = (
            df_eda
            .groupby("targets", as_index=False)
            .sample(n=num_of_samples, random_state=42)
            .reset_index(drop=True)
        )

        return df_one_per_target

    def save_eda_crop_target(self, df_eda, target_name, save_path, num_of_examples):
        df_eda = df_eda[df_eda["targets"] == target_name]
        df_eda = df_eda.sample(n=num_of_examples, random_state=42)
        for i, row in df_eda.iterrows():

            # get details
            img_name     = row["jpg_file"]
            bb           = row["bb"]
            img          = Image.open(f'{self.dataset_path}/{img_name}').convert("RGB")
            img_w, img_h = img.size

            # bbox conversion
            x1, y1, x2, y2 = self._yolo_to_xyxy(bb, img_w, img_h)
            crop     = self._crop_with_margin(img, (x1, y1, x2, y2), self.MARGIN)
            crop.save(f"{save_path}/{img_name}", "JPEG")



    def draw_eda(self, plot_x_size, plot_y_size, df_eda, label_column, save_path):

        # ---- PLOT 3x3 GRID ----
        fig, axes = plt.subplots(plot_x_size, plot_y_size, figsize=(12, 12))
        axes = axes.flatten()

        for i, row in df_eda.iterrows():
            img_path = row["jpg_file"]
            bb = row["bb"]
            if label_column == 'targets':
                label = row["targets"]
            elif label_column == 'jpg_file':
                label = row["jpg_file"]
            elif label_column == "family":
                label = f"{row['targets']} ({row['jpg_file']})"
            else:
                label = f"{row['targets']}_{row['jpg_file']}"

            img = Image.open(f'{self.dataset_path}/{img_path}').convert("RGB")
            img_w, img_h = img.size

            # bbox conversion
            x1, y1, x2, y2 = self._yolo_to_xyxy(bb, img_w, img_h)

            # crop with margin
            crop = self._crop_with_margin(img, (x1, y1, x2, y2), self.MARGIN)

            ax = axes[i]
            ax.imshow(crop)
            ax.set_title(str(label), fontsize=10)
            ax.axis("off")

        # hide unused subplots (if any)
        for j in range(len(df_eda), (plot_x_size * plot_y_size)):
            axes[j].axis("off")

        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def draw_eda_family(self, family_name, save_path):
        df_eda                = self.df_eda[self.df_eda["family"] == family_name]
        num_different_targets = len(set(df_eda["targets"]))
        NUM_OF_SAMPLES        = 4
        df_eda = self._get_targets_sub_sample(df_eda, num_of_samples=NUM_OF_SAMPLES)
        self.draw_eda(num_different_targets, NUM_OF_SAMPLES, df_eda, "family", save_path)

    def draw_eda_target(self, df_eda, target_name, save_path):
        df_eda = df_eda[df_eda["targets"] == target_name]
        df_eda = df_eda.sample(n=36, random_state=50).reset_index(drop=True)
        self.draw_eda(6, 6, df_eda, "jpg_file", save_path)

    def draw_eda_different_targets(self, df_eda, save_path):

        df_eda = self._get_targets_sub_sample(df_eda, num_of_samples=1)
        self.draw_eda(3, 3, df_eda, label_column="targets", save_path=save_path)



if __name__ == "__main__":

    df_eda         = pd.read_pickle("Pickles/train_db.pkl")
    edaDrawTargets = EdaDrawTargets(margin=150,
                                    dataset_path="/home/amitli/datasets/DOR_6/Train_B/Database/Images",
                                    df_eda=df_eda)

    family_name = "Anti aircraft"
    #family_name = "Launchers"
    # family_name = "Tank"


    edaDrawTargets.draw_eda_family(family_name, save_path=None)