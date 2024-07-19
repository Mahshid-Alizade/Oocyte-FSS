import torchvision.transforms.functional as TF
from torchvision import transforms
import pandas as pd
import openpyxl
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os, shutil
import random
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import MeanIoU

# np.set_printoptions(threshold=sys.maxsize)


def calculate_multi_iou(testlabel, pred):

    keras_iou = MeanIoU(num_classes=4)
    keras_iou.update_state(testlabel, pred)
    values = np.array(keras_iou.get_weights()).reshape(4, 4)
    class1_IoU = values[0, 0] / (
        values[0, 0]
        + values[0, 1]
        + values[0, 2]
        + values[0, 3]
        + values[1, 0]
        + values[2, 0]
        + values[3, 0]
    )
    class2_IoU = values[1, 1] / (
        values[1, 1]
        + values[1, 0]
        + values[1, 2]
        + values[1, 3]
        + values[0, 1]
        + values[2, 1]
        + values[3, 1]
    )
    class3_IoU = values[2, 2] / (
        values[2, 2]
        + values[2, 0]
        + values[2, 1]
        + values[2, 3]
        + values[0, 2]
        + values[1, 2]
        + values[3, 2]
    )
    class4_IoU = values[3, 3] / (
        values[3, 3]
        + values[3, 0]
        + values[3, 1]
        + values[3, 2]
        + values[0, 3]
        + values[1, 3]
        + values[2, 3]
    )

    # print("IoU for class1 is: ", class1_IoU)
    # print("IoU for class2 is: ", class2_IoU)
    # print("IoU for class3 is: ", class3_IoU)
    # print("IoU for class4 is: ", class4_IoU)

    # background iou is excluded
    return (class1_IoU, class2_IoU, class3_IoU)


def maskimg(img, mask, edge, color=[0, 0, 255], alpha=0.2, method="binary"):
    """
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [, , _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1].

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    """

    label_of_classes = np.unique(mask)
    out = img.copy()
    img_layer = img.copy()

    if method == "binary":
        alpha = 0.5
        img_layer[mask == 255] = color
        edge_layer = img.copy()
        edge_layer[edge == 255] = color
        out = cv2.addWeighted(edge_layer, 1, out, 0, 0, out)
        out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

    else:
        label_of_classes = label_of_classes[0:3]
        for i in label_of_classes:

            img_layer[mask == i] = color
            out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

        edge_layer = out.copy()
        edge_layer[edge == 255] = [0, 0, 0]
        out = cv2.addWeighted(edge_layer, 1, out, 0, 0, out)

    return out


def test_visualization(batches, batch_labels, stick, result_path, cnt, method="binary"):
    test_gt_label = np.transpose(batch_labels.numpy()[0], (1, 2, 0))
    test_gt_label = (test_gt_label * 255).astype(np.uint8)
    test_gt_edge = cv2.Canny(test_gt_label, 1, 1)

    testimg = np.transpose(batches.numpy()[0][0:3], (1, 2, 0))[:, :, ::-1] * 255
    testlabel = stick[224 * 3 : 224 * 4, 0:224, :].astype(np.uint8)
    testedge = cv2.Canny(testlabel, 1, 1)

    if method == "binary":
        test_gt_label = test_gt_label.copy()[:, :, 0]
    else:
        test_gt_label = np.argmax(test_gt_label, axis=2).copy()

    cv2.imwrite(
        result_path + "/test%s_gt.png" % cnt,
        maskimg(testimg, test_gt_label, test_gt_edge, color=[0, 255, 0], method=method),
    )
    cv2.imwrite(
        result_path + "/test%s.png" % cnt,
        maskimg(testimg, testlabel.copy()[:, :, 0], testedge, method=method),
    )


def train_multi_visualization(
    DISPLAY_QUERY, BATCH_NUM_PER_CLASS, output, batches, batch_labels
):
    colormap = {0: [0, 0, 0], 1: [0, 0, 255], 2: [255, 0, 100], 3: [255, 255, 255]}
    query_output = np.zeros((224 * 3, 224 * DISPLAY_QUERY, 3), dtype=np.uint8)
    chosen_query = random.sample(
        list(range(0, BATCH_NUM_PER_CLASS)), DISPLAY_QUERY
    )  # [0,1,2,3,4]

    # ---- query output ----
    for cnt, x in enumerate(chosen_query):
        # --- 1st row ---
        query_img = (np.transpose(batches.numpy()[x], (1, 2, 0)) * 255).astype(
            np.uint8
        )[:, :, :3][:, :, ::-1]
        query_output[0:224, cnt * 224 : (cnt + 1) * 224, :] = query_img
        # --- 2nd row ---
        query_label = batch_labels.numpy()[x]
        query_label = np.argmax(query_label, axis=0)
        rgb_label = class_mask_to_rgb(query_label, colormap)
        query_output[224 : 224 * 2, cnt * 224 : (cnt + 1) * 224, :] = rgb_label
        # --- 3rd row ---
        query_pred = output.detach().cpu().numpy()[x]
        query_pred = np.argmax(query_pred, axis=0)
        rgb_pred = class_mask_to_rgb(query_pred, colormap)
        query_output[224 * 2 : 224 * 3, cnt * 224 : (cnt + 1) * 224, :] = rgb_pred

    return query_output


def train_binary_visualization(
    DISPLAY_QUERY, BATCH_NUM_PER_CLASS, output, batches, batch_labels
):
    pred_colormap = {0: [0, 0, 0], 1: [255, 255, 255]}
    gt_colormap = {0: [0, 0, 0], 1: [34, 139, 34]}
    query_output = np.zeros((224 * 3, 224 * DISPLAY_QUERY, 3), dtype=np.uint8)
    chosen_query = random.sample(list(range(0, BATCH_NUM_PER_CLASS)), DISPLAY_QUERY)
    # ---- query output ----
    for cnt, x in enumerate(chosen_query):
        # --- 1st row ---
        query_img = (np.transpose(batches.numpy()[x], (1, 2, 0)) * 255).astype(
            np.uint8
        )[:, :, :3][:, :, ::-1]
        query_output[0:224, cnt * 224 : (cnt + 1) * 224, :] = query_img
        # --- 2nd row ---
        query_label = batch_labels.numpy()[x][0]
        rgb_label = class_mask_to_rgb(query_label, gt_colormap)
        # query_label[query_label != 0] = 1
        # query_label = decode_segmap(query_label)
        query_output[224 : 224 * 2, cnt * 224 : (cnt + 1) * 224, :] = rgb_label
        # --- 3rd row ---
        query_pred = output.detach().cpu().numpy()[x][0]
        # print(query_pred)
        # query_pred = (query_pred * 255).astype(np.uint8)
        query_pred[query_pred <= 0.5] = 0
        query_pred[query_pred > 0.5] = 1
        rgb_pred = class_mask_to_rgb(query_pred, pred_colormap)
        # result = np.zeros((224, 224, 3), dtype=np.uint8)
        # result[:, :, 0] = query_pred
        # result[:, :, 1] = query_pred
        # result[:, :, 2] = query_pred
        query_output[224 * 2 : 224 * 3, cnt * 224 : (cnt + 1) * 224, :] = rgb_pred

    return query_output


def class_mask_to_rgb(mask, colormap):
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_index, color in colormap.items():
        rgb_image[mask == class_index] = color

    return rgb_image


def rgb_to_class(mask_rgb):
    unique_colors = np.unique(mask_rgb.reshape(-1, mask_rgb.shape[2]), axis=0)
    class_map = {tuple(color): idx for idx, color in enumerate(unique_colors)}
    class_mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.int32)
    for color, idx in class_map.items():
        class_mask[(mask_rgb == color).all(axis=2)] = idx
    return class_mask, class_map


def empty_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def clean_dataset():

    folders = [
        "./fewshot-train/multi/",
        "./fewshot-train/image/",
        "./fewshot-train/label/o/",
        "./fewshot-train/label/p/",
        "./fewshot-train/label/z/",
        "./fewshot-test/query/image/",
        "./fewshot-test/query/label-multi/",
        "./fewshot-test/query/label-o/",
        "./fewshot-test/query/label-p/",
        "./fewshot-test/query/label-z/",
        "./fewshot-test/support/image/",
        "./fewshot-test/support/label-multi/",
        "./fewshot-test/support/label-o/",
        "./fewshot-test/support/label-p/",
        "./fewshot-test/support/label-z/",
    ]

    for i in folders:
        empty_folder(i)

    print("last dataset got cleaned successfully!!")


def save_img(path, img_name, image):
    cv2.imwrite(path + img_name, image)


def load_img(path, img_name):
    return cv2.imread(path + img_name)


def write_file(value="", file_name="convert.txt"):
    with open(file_name, "w") as convert_file:
        convert_file.write(json.dumps(value))


def read_file(file_name="convert.txt"):
    with open(file_name) as f:
        data = f.read()

    js = json.loads(data)
    return js


def save_result(
    columns=["episode", "chosen class", "loss", "train class"],
    rows=[],
    path="./train-result.xlsx",
):
    df = pd.read_excel(path)
    new_data = {}
    for i in range(len(columns)):
        new_data[columns[i]] = rows[i]

    df2 = pd.DataFrame(new_data)

    if df.empty:
        df2.to_excel(path)

    else:
        df = pd.concat([df, df2]).reset_index(drop=True)
        df.to_excel(path, columns=columns)


def reset_result(path="./", file_name="train-result"):
    workbook = openpyxl.load_workbook(path + file_name + ".xlsx")
    sheets = workbook.get_sheet_names()
    std = workbook.get_sheet_by_name(sheets[0])
    workbook.remove_sheet(std)
    workbook.create_sheet()
    workbook.save(path + file_name + ".xlsx")


def previous_run(path):

    with open(path, "r") as file:
        content_list = file.readlines()
        chosen_img = [line.strip() for line in content_list]

    return chosen_img


def save_pictures(chosen_img, path):

    # saving chosen pictures for next run
    with open(path, "w") as file:
        for item in chosen_img:
            file.write(f"{item}\n")


# binary segmentation : num_classes = 2, ignore_index = 0
# multi segmentation : num_classes = 4, ignore_index = 3
def perf_measure(target, pred, num_classes=2, ignore_index=0):
    """
    Calculate TP, TN, FP, FN for multi-label segmentation.

    Parameters:
    pred (np.ndarray): Prediction array of shape (height, width).
    target (np.ndarray): Ground truth array of shape (height, width).
    num_classes (int): Number of classes including the background.
    ignore_index (int): Index of the background class to ignore.

    Returns:
    dict: A dictionary with TP, TN, FP, FN for each class.
    """
    tp = np.array([])
    tn = np.array([])
    fp = np.array([])
    fn = np.array([])

    # Flatten the arrays
    pred = pred.flatten()
    target = target.flatten()

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        # True Positives: both pred and target are the current class
        tp = np.append(tp, np.sum((pred == cls) & (target == cls)))

        # True Negatives: both pred and target are not the current class
        tn = np.append(tn, np.sum((pred != cls) & (target != cls)))

        # False Positives: pred is the current class but target is not
        fp = np.append(
            fp, np.sum((pred == cls) & (target != cls))  # & (target != ignore_index)
        )

        # False Negatives: target is the current class but pred is not
        fn = np.append(fn, np.sum((pred != cls) & (target == cls)))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def precision(TP, FP):
    return TP / (TP + FP)


def recall(TP, FN):
    return TP / (TP + FN)


def dsc(TP, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN)
