from PIL      import Image, ImageDraw, ImageFont
import numpy as np

from app_config.settings import FONT_FILE


class BoundingBoxMatcher:
    """
    An advanced utility class to match predicted bounding boxes from Gemma4
    with Ground Truth (GT) bounding boxes using IoU or IoG (Intersection over GT).
    """

    def __init__(self, threshold: float = 0.6, criterion: str = "iou"):
        """
        Initializes the matcher.

        :param threshold: Minimum score value required to consider a match valid.
        :param criterion: 'iou' for Intersection over Union,
                          'iog' for Intersection over Ground Truth (Coverage).
        """
        self.threshold = threshold
        self.criterion = criterion.lower()
        if self.criterion not in ["iou", "iog"]:
            raise ValueError("Criterion must be either 'iou' or 'iog'")

    @staticmethod
    def _model_to_pascal_voc(box: list) -> list:
        """Converts [ymin, xmin, ymax, xmax] (0-1000) to [xmin, ymin, xmax, ymax] (0-1)"""
        ymin, xmin, ymax, xmax = box
        return [xmin / 1000.0, ymin / 1000.0, xmax / 1000.0, ymax / 1000.0]

    @staticmethod
    def _gt_to_pascal_voc(box: list) -> list:
        """Converts YOLO [x_center, y_center, width, height] (0-1) to [xmin, ymin, xmax, ymax] (0-1)"""
        x_center, y_center, width, height = box
        xmin = x_center - (width / 2.0)
        ymin = y_center - (height / 2.0)
        xmax = x_center + (width / 2.0)
        ymax = y_center + (height / 2.0)
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def _calculate_metrics(box1: list, box2: list) -> tuple:
        """
        Calculates both IoU and IoG for box1 (prediction) and box2 (ground truth).
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0.0
        iog = intersection_area / box2_area if box2_area > 0 else 0.0

        return iou, iog

    def match(self, l_model_bb: list, l_gt_bb: list) -> list:
        """
        Matches each bounding box from l_model_bb to its best corresponding box in l_gt_bb.
        """
        if len(l_gt_bb) > 0:
            if type(l_gt_bb[0][0]) == str:
                l_gt_bb = [[float(x) for x in row] for row in l_gt_bb]

        model_boxes_std = [self._model_to_pascal_voc(box) for box in l_model_bb]
        gt_boxes_std = [self._gt_to_pascal_voc(box) for box in l_gt_bb]

        results = []

        for m_idx, m_box in enumerate(model_boxes_std):
            best_gt_idx = -1
            max_score = 0.0

            for g_idx, g_box in enumerate(gt_boxes_std):
                iou, iog = self._calculate_metrics(m_box, g_box)
                score = iou if self.criterion == "iou" else iog

                if score > max_score:
                    max_score = score
                    best_gt_idx = g_idx

            if max_score >= self.threshold:
                results.append({
                    "model_index": m_idx,
                    "matched_gt_index": best_gt_idx,
                    "score": max_score
                })
            else:
                results.append({
                    "model_index": m_idx,
                    "matched_gt_index": -1,
                    "score": max_score
                })

        return results



def draw_result(full_path, l_gt_bb, l_model_bb):


    img        = Image.open(full_path)
    font       = ImageFont.truetype(FONT_FILE, size=16)
    draw       = ImageDraw.Draw(img)
    img_w, img_h = img.size

    # add GT
    for i in range(len(l_gt_bb)):
        x_center, y_center, bw, bh = l_gt_bb[i]


        # Convert to pixel coordinates
        x_center *= img_w
        y_center *= img_h
        bw *= img_w
        bh *= img_h

        x1 = x_center - bw / 2
        y1 = y_center - bh / 2
        x2 = x_center + bw / 2
        y2 = y_center + bh / 2

        draw.rectangle([x1, y1, x2, y2], outline="green", width=1)
        draw.text((x1 + 50, y1), f"GT: {i}", fill="green", font=font)


    for i in range(len(l_model_bb)):
        ymin, xmin, ymax, xmax = l_model_bb[i]

        # 2. Convert from normalized (0-1000) to actual pixel values
        left = xmin * img_w / 1000
        top = ymin * img_h / 1000
        right = xmax * img_w / 1000
        bottom = ymax * img_h / 1000

        draw.rectangle([left, top, right, bottom], outline="red", width=1)
        draw.text((left + 10, top+10), f"PRED: {i}", fill="red", font=font)

    img.show()


if __name__ == "__main__":
    l_model_bb = [[448, 473, 536, 493], [512, 445, 546, 467], [400, 546, 474, 571], [502, 531, 562, 554]]
    l_gt_bb = [[0.503881, 0.406096, 0.0375104, 0.0313761], [0.481812, 0.485661, 0.0149696, 0.0695401]]

    # l_model_bb = [[448, 473, 536, 493]]
    # l_gt_bb = [[0.481812, 0.485661, 0.0149696, 0.0695401]]

    full_image_path = '/home/amitli/datasets/DOR_6/Train_B/Database/Images/2_844400_1288_09-56-49.jpg'
    #draw_result(full_image_path, l_gt_bb, l_model_bb)
    #exit(0)

    matcher = BoundingBoxMatcher(threshold=0.9, criterion="iog")
    matching_results = matcher.match(l_model_bb, l_gt_bb)
    for matcing in matching_results:
        score = matcing["score"]
        match_gt_index = matcing["matched_gt_index"]
        model_index = matcing["model_index"]
        print(f"Model Index: {model_index}, Matched GT Index: {match_gt_index}, score = {score:.2f}")
