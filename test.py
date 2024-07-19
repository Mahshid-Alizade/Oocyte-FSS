import torch
import numpy as np
import os
import argparse
import cv2
import subprocess
import matplotlib.pyplot as plt
from utils import *
from network import *

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from tensorflow.keras.utils import to_categorical


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=1)
parser.add_argument("-ln", "--label_num", type=int, default=4)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=1)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-ism", "--ImageSelectionMethod", type=str, default="binary")


parser.add_argument(
    "-modelf",
    "--feature_encoder_model",
    type=str,
    default=r"trained-models\20000_binary_FE.pkl",
    # "pretrained-models\FE_pretrained.pkl",
)
parser.add_argument(
    "-modelr",
    "--relation_network_model",
    type=str,
    default=r"trained-models\20000_binary_RN.pkl",
    # "pretrained-models\RN_pretrained.pkl",
)
parser.add_argument(
    "-sld", "--support_label_dir", type=str, default="fewshot-test\support\label-z"
)
parser.add_argument(
    "-sid",
    "--support_image_dir",
    type=str,
    default="fewshot-test\support\image",
)
parser.add_argument(
    "-tld",
    "--test_label_dir",
    type=str,
    default=r".\fewshot-test\query\label-z",
)
parser.add_argument(
    "-tid",
    "--test_image_dir",
    type=str,
    default=r".\fewshot-test\query\image",
)

args = parser.parse_args()

# Run the 'nvidia-smi' command to get GPU memory information
nvidia_smi_output = subprocess.Popen(
    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
    shell=True,
    stdout=subprocess.PIPE,
).stdout.readlines()

# Convert the output to a list of integers
free_memory = [int(x.strip()) for x in nvidia_smi_output]

# Find the index of the GPU with the most available memory
best_gpu_index = free_memory.index(max(free_memory))

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
LABEL_NUM = args.label_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
GPU = args.gpu
FEATURE_MODEL = args.feature_encoder_model
RELATION_MODEL = args.relation_network_model
METHOD = args.ImageSelectionMethod

assert METHOD == "binary" or METHOD == "multi", "METHOD must be chosen binary or multi"


def get_oneshot_multi_batch(test_img_name, test_lbl_name):

    support_images = np.zeros(
        (CLASS_NUM * SAMPLE_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32
    )
    support_labels = np.zeros(
        (CLASS_NUM * SAMPLE_NUM_PER_CLASS, LABEL_NUM, 224, 224), dtype=np.float32
    )
    query_images = np.zeros(
        (CLASS_NUM * BATCH_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32
    )
    query_labels = np.zeros(
        (CLASS_NUM * BATCH_NUM_PER_CLASS, LABEL_NUM, 224, 224), dtype=np.float32
    )
    zeros = np.zeros(
        (CLASS_NUM * BATCH_NUM_PER_CLASS, LABEL_NUM, 224, 224), dtype=np.float32
    )

    # support_images
    imgnames = os.listdir(".\%s" % args.support_image_dir)
    lblnames = os.listdir(".\%s" % args.support_label_dir)
    chosen_index = list(range(0, SAMPLE_NUM_PER_CLASS))

    for k in chosen_index:
        # process support images
        image = cv2.imread("%s\%s" % (args.support_image_dir, imgnames[k]))
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))

        label = cv2.imread("%s\%s" % (args.support_label_dir, lblnames[k]))

        mask_rgb = np.array(label)
        class_mask, class_map = rgb_to_class(mask_rgb)
        unique_values = np.unique(class_mask)
        num_classes = len(unique_values)

        resized_class_mask = cv2.resize(
            class_mask, (224, 224), interpolation=cv2.INTER_NEAREST
        )
        resized_class_mask = np.expand_dims(resized_class_mask, axis=-1)
        categorical_mask = to_categorical(resized_class_mask, num_classes=num_classes)
        label = np.transpose(categorical_mask, (2, 0, 1))

        support_images[k] = image
        support_labels[k] = label

    # process query images
    testimage = cv2.imread("%s\%s" % (args.test_image_dir, test_img_name))
    testimage = cv2.resize(testimage, (224, 224))
    testimage = testimage / 255.0
    testimage = np.transpose(testimage, (2, 0, 1))

    testlabel = cv2.imread("%s\%s" % (args.test_label_dir, test_lbl_name))
    mask_rgb = np.array(testlabel)
    class_mask, class_map = rgb_to_class(mask_rgb)
    unique_values = np.unique(class_mask)
    num_classes = len(unique_values)

    resized_class_mask = cv2.resize(
        class_mask, (224, 224), interpolation=cv2.INTER_NEAREST
    )
    resized_class_mask = np.expand_dims(resized_class_mask, axis=-1)
    categorical_mask = to_categorical(resized_class_mask, num_classes=num_classes)
    testlabel = np.transpose(categorical_mask, (2, 0, 1))

    query_images[0] = testimage
    query_labels[0] = testlabel

    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat(
        (support_images_tensor, support_labels_tensor), dim=1
    )

    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return (
        support_images_tensor,
        support_labels_tensor,
        query_images_tensor,
        query_labels_tensor,
    )


def get_oneshot_batch(test_img_name, test_lbl_name):

    support_images = np.zeros(
        (CLASS_NUM * SAMPLE_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32
    )
    support_labels = np.zeros(
        (CLASS_NUM * SAMPLE_NUM_PER_CLASS, CLASS_NUM, 224, 224), dtype=np.float32
    )
    query_images = np.zeros(
        (CLASS_NUM * BATCH_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32
    )
    query_labels = np.zeros(
        (CLASS_NUM * BATCH_NUM_PER_CLASS, CLASS_NUM, 224, 224), dtype=np.float32
    )
    zeros = np.zeros((CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 224, 224), dtype=np.float32)

    # support_images
    imgnames = os.listdir(".\%s" % args.support_image_dir)
    lblnames = os.listdir(".\%s" % args.support_label_dir)
    chosen_index = list(range(0, SAMPLE_NUM_PER_CLASS))  # [0:5]

    for k in chosen_index:
        # process support images
        image = cv2.imread(
            "%s\%s" % (args.support_image_dir, imgnames[k]), cv2.IMREAD_COLOR
        )
        image = cv2.resize(image, (224, 224))
        image = image / 255.0

        label = cv2.imread("%s\%s" % (args.support_label_dir, lblnames[k]))[:, :, 0]
        label = cv2.resize(label, (224, 224))
        label = label / 255.0
        label = np.expand_dims(label, axis=-1)

        image = np.transpose(image, (2, 1, 0))
        label = np.transpose(label, (2, 1, 0))

        support_images[k] = image
        support_labels[k][0] = label

    # process query images
    testimage = cv2.imread(
        "%s\%s" % (args.test_image_dir, test_img_name), cv2.IMREAD_COLOR
    )
    testimage = cv2.resize(testimage, (224, 224))
    testimage = testimage / 255.0

    testlabel = cv2.imread("%s\%s" % (args.test_label_dir, test_lbl_name))[:, :, 0]
    testlabel = cv2.resize(testlabel, (224, 224))
    testlabel = testlabel / 255.0
    testlabel = np.expand_dims(testlabel, axis=-1)

    testimage = np.transpose(testimage, (2, 1, 0))
    testlabel = np.transpose(testlabel, (2, 1, 0))

    query_images[0], query_labels[0] = (
        testimage,
        testlabel,
    )  # resize(testimage, testlabel)

    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat(
        (support_images_tensor, support_labels_tensor), dim=1
    )

    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return (
        support_images_tensor,
        support_labels_tensor,
        query_images_tensor,
        query_labels_tensor,
    )


def main():
    # Step 2: init neural networks
    print("init neural networks")

    if METHOD == "multi":
        input_channel = 7
        output_channel = 4
    else:
        input_channel = 4
        output_channel = 1

    feature_encoder = CNNEncoder(input_channel)
    relation_network = RelationNetwork(output_channel)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    if os.path.exists(FEATURE_MODEL):
        feature_encoder.load_state_dict(torch.load(FEATURE_MODEL))
        print("load feature encoder success")
    else:
        raise Exception("Can not load feature encoder: %s" % FEATURE_MODEL)

    if os.path.exists(RELATION_MODEL):
        relation_network.load_state_dict(torch.load(RELATION_MODEL))
        print("load relation network success")
    else:
        raise Exception("Can not load relation network: %s" % RELATION_MODEL)

    print("Testing...")

    if os.path.exists("test-result"):
        os.system("rmdir -r test-result")
    if not os.path.exists("test-result"):
        os.makedirs("test-result")
    if not os.path.exists("./test-result/"):
        os.makedirs("./test-result/")

    stick = np.zeros((224 * 4, 224 * 5, 3), dtype=np.uint8)
    classname = args.support_label_dir.split("\\")[-1]

    if METHOD == "multi":
        number_of_classes = 3
        classes = {0: "O", 1: "P", 2: "Z"}
    else:
        number_of_classes = 1
        classes = {0: classname}

    perf_measures = {}
    TP, TN, FP, FN = [], [], [], []
    iou = np.zeros(number_of_classes)
    classiou = np.zeros(number_of_classes)
    meanaccuracy = np.zeros(number_of_classes)
    meanprecision = np.zeros(number_of_classes)
    meanrecall = np.zeros(number_of_classes)
    meanDSC = np.zeros(number_of_classes)

    test_img_names = os.listdir("%s" % args.test_image_dir)
    test_lbl_names = os.listdir("%s" % args.test_label_dir)

    print("%s testing images in class %s" % (len(test_img_names), classname))

    result_path = "./test-result/" + classname
    result_file = result_path + "/IOU_results.txt"
    open(result_file, "w").close()

    for cnt, test_img_name in enumerate(test_img_names):
        cnt = cnt + 1
        if METHOD == "multi":
            (samples, sample_labels, batches, batch_labels) = get_oneshot_multi_batch(
                test_img_name, test_lbl_names[cnt - 1]
            )
        else:
            (samples, sample_labels, batches, batch_labels) = get_oneshot_batch(
                test_img_name, test_lbl_names[cnt - 1]
            )

        output = calculate_output(
            samples,
            batches,
            feature_encoder,
            relation_network,
            CLASS_NUM,
            METHOD,
            SAMPLE_NUM_PER_CLASS,
            BATCH_NUM_PER_CLASS,
            GPU,
        )

        for i in range(0, batches.size()[0]):
            # get prediction
            if METHOD == "multi":
                demo = np.zeros((224, 224, 3))
                pred = output.data.cpu().numpy()[i]
                testlabel = batch_labels.numpy()[i]
                pred = np.argmax(pred, axis=0)
                testlabel = np.argmax(testlabel, axis=0)

                demo[:, :, 0] = pred
                demo[:, :, 1] = pred
                demo[:, :, 2] = pred
                stick[224 * 3 : 224 * 4, 224 * i : 224 * (i + 1), :] = demo.copy()

                # compute IOU
                iou = calculate_multi_iou(testlabel, pred)  # [-,-,-]
                perf_measures = perf_measure(
                    testlabel, pred, num_classes=number_of_classes + 1, ignore_index=3
                )
                TP, TN, FP, FN = (
                    perf_measures["TP"],
                    perf_measures["TN"],
                    perf_measures["FP"],
                    perf_measures["FN"],
                )

            else:
                pred = output.data.cpu().numpy()[i][0]
                pred[pred <= 0.5] = 0
                pred[pred > 0.5] = 1
                demo = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB) * 255
                stick[224 * 3 : 224 * 4, 224 * i : 224 * (i + 1), :] = demo.copy()
                testlabel = batch_labels.numpy()[i][0].astype(bool)
                pred = pred.astype(bool)

                # compute IOU
                overlap = testlabel * pred
                union = testlabel + pred
                iou[0] = overlap.sum() / float(union.sum())
                perf_measures = perf_measure(testlabel, pred)
                TP, TN, FP, FN = (
                    perf_measures["TP"],
                    perf_measures["TN"],
                    perf_measures["FP"],
                    perf_measures["FN"],
                )

            for i in range(number_of_classes):
                classiou[i] += iou[i]
                meanaccuracy[i] += accuracy(TP[i], TN[i], FP[i], FN[i])
                meanprecision[i] += precision(TP[i], FP[i])
                meanrecall[i] += recall(TP[i], FN[i])
                meanDSC[i] += dsc(TP[i], FP[i], FN[i])

                with open(result_file, "a") as file:
                    file.write(
                        "%s / %s IOU %s = %0.4f"
                        % (cnt, len(test_img_names), classes[i], iou[i])
                    )
                    file.write("\n")
                print(
                    "%s / %s" % (cnt, len(test_img_names)),
                    " IOU %s = %0.4f" % (classes[i], iou[i]),
                )
        # visulization
        if cnt == 1:
            for i in range(0, samples.size()[0]):
                suppimg = (
                    np.transpose(samples.numpy()[i][0:3], (1, 2, 0))[:, :, ::-1] * 255
                )
                supplabel = np.transpose(sample_labels.numpy()[i], (1, 2, 0))
                # if METHOD == "binary":
                #     supplabel = cv2.cvtColor(supplabel, cv2.COLOR_GRAY2RGB)
                supplabel = (supplabel * 255).astype(np.uint8)
                suppedge = cv2.Canny(supplabel, 1, 1)
                if METHOD == "binary":
                    supplabel = supplabel.copy()[:, :, 0]
                else:
                    supplabel = np.argmax(supplabel, axis=2).copy()

                cv2.imwrite(
                    result_path + "/support_%s.png" % (i + 1),
                    maskimg(
                        suppimg,
                        supplabel,
                        suppedge,
                        color=[0, 255, 0],
                        method=METHOD,
                    ),
                )

        test_visualization(
            batches, batch_labels, stick, result_path, cnt, method=METHOD
        )

    for i in range(number_of_classes):
        classiou[i] /= len(test_img_names)
        meanaccuracy[i] /= len(test_img_names)
        meanprecision[i] /= len(test_img_names)
        meanrecall[i] /= len(test_img_names)
        meanDSC[i] /= len(test_img_names)

    print(
        "\nClass name: ", classname, " Tested on model: ", RELATION_MODEL.split("\\")[1]
    )
    print("The average IOU: ", classiou)
    print("The average accuracy: ", meanaccuracy)
    print("The average precision: ", meanprecision)
    print("The average recall: ", meanrecall)
    print("The average dsc: ", meanDSC)

    with open(result_file, "a") as file:
        file.write(
            "\nClass name: %s Tested on model: %s"
            % (classname, RELATION_MODEL.split("\\")[1])
        )
    with open(result_file, "a") as file:
        file.write("\nThe average IOU: %s" % classiou)
    with open(result_file, "a") as file:
        file.write("\nThe average accuracy: %s" % meanaccuracy)
    with open(result_file, "a") as file:
        file.write("\nThe average precision: %s" % meanprecision)
    with open(result_file, "a") as file:
        file.write("\nThe average recall: %s" % meanrecall)
    with open(result_file, "a") as file:
        file.write("\nThe average dsc: %s" % meanDSC)


if __name__ == "__main__":
    # reset_result()
    main()
