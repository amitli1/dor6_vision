from app_config.settings import FONT_FILE, TEST_CROP_FILES_PATH
from pointing_agent.pointing_agent import PointingAgent
from weapon_system_classification.weapon_system_classification import WeaponSystemClassification
from tqdm import tqdm
import glob
import os
import pandas as pd

def run_full_pipeline(wsc, pointingAgent):
    # l_files = glob.glob('/home/amitli/repo/dor6_vision/Dataset/test_set/*.jpg')
    l_files = glob.glob('/home/amitli/repo/dor6_vision/Dataset/test_set_crop/*.jpg')
    l_point_pred = []
    l_point_x = []
    l_point_y = []
    l_classifcation_pred = []
    l_jpg_files = []

    for file in tqdm(l_files):
        point_pred, x, y = pointingAgent.run_molmo_prediction(file)
        classifcation_pred = wsc.get_per_class_confidence(file)

        l_jpg_files.append(os.path.basename(file))
        l_point_pred.append(point_pred)
        l_point_x.append(x)
        l_point_y.append(y)
        l_classifcation_pred.append(classifcation_pred)

    df = pd.DataFrame({"jpg_file": l_jpg_files,
                       "point_pred": l_point_pred,
                       "point_x": l_point_x,
                       "point_y": l_point_y,
                       "classification_pred": l_classifcation_pred})
    df.to_csv("/home/amitli/repo/dor6_vision/results/test_set_crop_prediction.csv", index=False)


if __name__ == '__main__':

    wsc = WeaponSystemClassification(use_knn=True)
    wsc.get_per_class_confidence(r'/home/amitli/repo/ball.jpg')

    TRAIN = False

    if TRAIN:
        pointingAgent = PointingAgent()
        pointingAgent.run_molmo_prediction(r'/home/amitli/repo/ball.jpg')

        run_full_pipeline(wsc, pointingAgent)

    df_test_crop         = pd.read_csv('/home/amitli/repo/dor6_vision/Dataset/test_set_point.csv')
    l_classifcation_pred = []
    for i in tqdm(range(len(df_test_crop))):
        jpg_file  = df_test_crop['jpg_file'].values[i]
        if jpg_file != 'frame_1170.jpg':
            continue
        crop_file = f"{TEST_CROP_FILES_PATH}/{jpg_file}"
        classifcation_pred = wsc.get_per_class_confidence(crop_file)
        l_classifcation_pred.append(classifcation_pred)

    df_results = pd.DataFrame({"jpg_file": df_test_crop['jpg_file'].values.tolist(),
                       "point_x": df_test_crop['x'].values.tolist(),
                       "point_y": df_test_crop['y'].values.tolist(),
                       "classification_pred": l_classifcation_pred})
    df_results.to_csv("/home/amitli/repo/dor6_vision/results/test_set_crop_prediction.csv", index=False)


