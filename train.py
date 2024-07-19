import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import random
import cv2
import matplotlib.pyplot as plt
import sys

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from utils import *
from network import *
from loss import *

# from evaluate import evaluate

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.utils import to_categorical

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-tipc", "--total_images_per_class", type=int, default=300)
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=1)
parser.add_argument("-ln", "--label_num", type=int, default=4)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)  #######
parser.add_argument("-maxS", "--maximun_support_num", type=int, default=10)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=5)
parser.add_argument("-d", "--display_query_num", type=int, default=5)
parser.add_argument("-start", "--start_episode", type=int, default=0)
parser.add_argument("-e", "--episode", type=int, default=20000)
# parser.add_argument("-lr", "--last_run", type=bool, default=False)
parser.add_argument("-lg", "--last_generate", type=bool, default=True)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-fi", "--finetune", type=bool, default=True)
parser.add_argument("-rff", "--ResultSaveFreq", type=int, default=1000)
parser.add_argument("-msf", "--ModelSaveFreq", type=int, default=1000)
parser.add_argument("-rf", "--TrainResultPath", type=str, default="train-result")
parser.add_argument("-msp", "--ModelSavePath", type=str, default="trained-models")
parser.add_argument("-ism", "--ImageSelectionMethod", type=str, default="binary")
parser.add_argument("-tdir", "--TrainDirectory", type=str, default="./fewshot-train/")
# parser.add_argument("-u", "--hidden_unit", type=int, default=10)
# parser.add_argument("-ex", "--exclude_class", type=int, default=6)
# parser.add_argument("-lo", "--loadImagenet", type=bool, default=False)
# parser.add_argument("-t", "--test_episode", type=int, default=1000)


parser.add_argument(
    "-modelf",
    "--feature_encoder_model",
    type=str,
    default="",  # r"trained-models\17000_binary_FE.pkl",
)
parser.add_argument(
    "-modelr",
    "--relation_network_model",
    type=str,
    default="",  # r"trained-models\17000_binary_RN.pkl",
)

parser.add_argument(
    "-fopt",
    "--FeatureOptim",
    type=str,
    default="",  # r"trained-models\optimizer\binary_FE_optimizer.pkl",
)
parser.add_argument(
    "-ropt",
    "--RelationOptim",
    type=str,
    default="",  # r"trained-models\optimizer\binary_RN_optimizer.pkl",
)

args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
LAST_GENERATE = args.last_generate
# LAST_RUN = args.last_run
CLASS_NUM = args.class_num
LABEL_NUM = args.label_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
SAVE_FREQ = args.ModelSaveFreq
# TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
# HIDDEN_UNIT = args.hidden_unit
DISPLAY_QUERY = args.display_query_num
# EXCLUDE_CLASS = args.exclude_class
FEATURE_MODEL = args.feature_encoder_model
RELATION_MODEL = args.relation_network_model
TOTAL_IMAGES_PER_CLASS = args.total_images_per_class
METHOD = args.ImageSelectionMethod
TRAIN_DIR = args.TrainDirectory
FEATURE_OPTIM = args.FeatureOptim
RELATION_OPTIM = args.RelationOptim
MAX_SUPPORT_NUM = args.maximun_support_num

assert METHOD == "binary" or METHOD == "multi", "METHOD must be chosen binary or multi"


def generate_dataset_folder_structure():

    all_imgs = os.listdir("./dataset/image/")

    train_size = 750 - MAX_SUPPORT_NUM
    test_size = 253 + MAX_SUPPORT_NUM

    train_imgs = random.sample(all_imgs, train_size)
    test_imgs = [x for x in all_imgs if x not in train_imgs]

    train_lbl_o = []
    train_lbl_p = []
    train_lbl_z = []
    train_lbl_multi = []

    test_lbl_o = []
    test_lbl_p = []
    test_lbl_z = []
    test_lbl_multi = []

    for i in range(train_size):
        train_lbl_o.append(train_imgs[i].split(".")[0] + "_%s" % "o" + ".png")
        train_lbl_p.append(train_imgs[i].split(".")[0] + "_%s" % "p" + ".png")
        train_lbl_z.append(train_imgs[i].split(".")[0] + "_%s" % "z" + ".png")
        train_lbl_multi.append(train_imgs[i].split(".")[0] + "_%s" % "multi" + ".png")

    for i in range(test_size):
        test_lbl_o.append(test_imgs[i].split(".")[0] + "_%s" % "o" + ".png")
        test_lbl_p.append(test_imgs[i].split(".")[0] + "_%s" % "p" + ".png")
        test_lbl_z.append(test_imgs[i].split(".")[0] + "_%s" % "z" + ".png")
        test_lbl_multi.append(test_imgs[i].split(".")[0] + "_%s" % "multi" + ".png")

    train_img_path = "./fewshot-train/image/"
    train_lbl_o_path = "./fewshot-train/label/o/"
    train_lbl_p_path = "./fewshot-train/label/p/"
    train_lbl_z_path = "./fewshot-train/label/z/"
    train_lbl_multi_path = "./fewshot-train/multi/"

    # saving training images
    for i in range(train_size):
        save_img(
            train_img_path,
            train_imgs[i],
            load_img(path="./dataset/image/", img_name=train_imgs[i]),
        )
        save_img(
            train_lbl_o_path,
            train_lbl_o[i],
            load_img("./dataset/label/o/", train_lbl_o[i]),
        )
        save_img(
            train_lbl_p_path,
            train_lbl_p[i],
            load_img("./dataset/label/p/", train_lbl_p[i]),
        )
        save_img(
            train_lbl_z_path,
            train_lbl_z[i],
            load_img("./dataset/label/z/", train_lbl_z[i]),
        )
        save_img(
            train_lbl_multi_path,
            train_lbl_multi[i],
            load_img("./dataset/multi/", train_lbl_multi[i]),
        )

    test_img_path_q = "./fewshot-test/query/image/"
    test_lbl_o_path_q = "./fewshot-test/query/label-o/"
    test_lbl_p_path_q = "./fewshot-test/query/label-p/"
    test_lbl_z_path_q = "./fewshot-test/query/label-z/"
    test_lbl_multi_path_q = "./fewshot-test/query/label-multi/"

    test_img_path_s = "./fewshot-test/support/image/"
    test_lbl_o_path_s = "./fewshot-test/support/label-o/"
    test_lbl_p_path_s = "./fewshot-test/support/label-p/"
    test_lbl_z_path_s = "./fewshot-test/support/label-z/"
    test_lbl_multi_path_s = "./fewshot-test/support/label-multi/"

    for i in range(test_size):
        # support
        if i < MAX_SUPPORT_NUM:
            save_img(
                test_img_path_s,
                test_imgs[i],
                load_img(path="./dataset/image/", img_name=test_imgs[i]),
            )
            save_img(
                test_lbl_o_path_s,
                test_lbl_o[i],
                load_img("./dataset/label/o/", test_lbl_o[i]),
            )
            save_img(
                test_lbl_p_path_s,
                test_lbl_p[i],
                load_img("./dataset/label/p/", test_lbl_p[i]),
            )
            save_img(
                test_lbl_z_path_s,
                test_lbl_z[i],
                load_img("./dataset/label/z/", test_lbl_z[i]),
            )
            save_img(
                test_lbl_multi_path_s,
                test_lbl_multi[i],
                load_img("./dataset/multi/", test_lbl_multi[i]),
            )
        # query
        else:
            save_img(
                test_img_path_q,
                test_imgs[i],
                load_img(path="./dataset/image/", img_name=test_imgs[i]),
            )
            save_img(
                test_lbl_o_path_q,
                test_lbl_o[i],
                load_img("./dataset/label/o/", test_lbl_o[i]),
            )
            save_img(
                test_lbl_p_path_q,
                test_lbl_p[i],
                load_img("./dataset/label/p/", test_lbl_p[i]),
            )
            save_img(
                test_lbl_z_path_q,
                test_lbl_z[i],
                load_img("./dataset/label/z/", test_lbl_z[i]),
            )
            save_img(
                test_lbl_multi_path_q,
                test_lbl_multi[i],
                load_img("./dataset/multi/", test_lbl_multi[i]),
            )


def load_multi_data(num=0):
    path = "./chosen-images/chosen_multi_imgs.txt"
    chosen_img = []

    if LAST_GENERATE == True:
        chosen_img = previous_run(path)

    else:
        imgs = os.listdir(TRAIN_DIR + "image/")
        chosen_img = random.sample(imgs, int(num))
        save_pictures(chosen_img, path)

    chosen_multilbl = []
    for i in range(num):
        chosen_multilbl.append(chosen_img[i].split(".")[0] + "_%s" % "multi" + ".png")

    return (chosen_img, chosen_multilbl)


def load_data(num=0, method="random"):
    chosen_img = []

    # if method == "random":
    path = "./chosen-images/chosen_imgs.txt"

    if LAST_GENERATE == True:
        chosen_img = previous_run(path)
    else:
        imgs = os.listdir(TRAIN_DIR + "image/")
        chosen_img = random.sample(imgs, int(num))
        save_pictures(chosen_img, path)

    chosen_lbl_o = []
    chosen_lbl_p = []
    chosen_lbl_z = []

    for i in range(num):
        chosen_lbl_o.append(chosen_img[i].split(".")[0] + "_%s" % "o" + ".png")
        chosen_lbl_p.append(chosen_img[i].split(".")[0] + "_%s" % "p" + ".png")
        chosen_lbl_z.append(chosen_img[i].split(".")[0] + "_%s" % "z" + ".png")

    return (
        (chosen_img, chosen_lbl_o),
        (chosen_img, chosen_lbl_p),
        (chosen_img, chosen_lbl_z),
    )
    # elif method == "iou":

    #     pass


def get_oneshot_batch(chosen_class_name="", chosen_image_label=()):
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

    # ['file (0).png, ...],  ['file (0)_o.png, ...]
    imgnames, lblnames = chosen_image_label

    # for i in range(len(chosen_class_name)):  # one class
    indexs = list(range(0, len(imgnames)))
    chosen_index = random.sample(indexs, SAMPLE_NUM_PER_CLASS + BATCH_NUM_PER_CLASS)

    j = 0
    for k in chosen_index:
        np.set_printoptions(threshold=sys.maxsize)
        img_path = TRAIN_DIR + "image/%s" % imgnames[k]
        lbl_path = TRAIN_DIR + "label/%s/%s" % (chosen_class_name, lblnames[k])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0

        label = cv2.imread(lbl_path)[:, :, 0]
        label = cv2.resize(label, (224, 224))
        label = label / 255.0
        label = np.expand_dims(label, axis=-1)

        image = np.transpose(image, (2, 1, 0))
        label = np.transpose(label, (2, 1, 0))

        if j < SAMPLE_NUM_PER_CLASS:
            support_images[j] = image
            support_labels[j][0] = label
        else:
            query_images[j - SAMPLE_NUM_PER_CLASS] = image
            query_labels[j - SAMPLE_NUM_PER_CLASS][0] = label
        j += 1

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


def get_oneshot_multi_batch(chosen_image_label=()):

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

    # ['file (0).png, ...],  ['file (0)_o.png, ...]
    imgnames, lblnames = chosen_image_label

    indexs = list(range(0, len(imgnames)))
    chosen_index = random.sample(indexs, SAMPLE_NUM_PER_CLASS + BATCH_NUM_PER_CLASS)

    j = 0
    for k in chosen_index:

        img_path = TRAIN_DIR + "image/%s" % imgnames[k]
        lbl_path = TRAIN_DIR + "multi/%s" % (lblnames[k])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0

        label = cv2.imread(lbl_path)

        mask_rgb = np.array(label)
        class_mask, class_map = rgb_to_class(mask_rgb)
        unique_values = np.unique(class_mask)
        num_classes = len(unique_values)

        # print("Unique values in the mask:", unique_values)
        # print("Number of classes:", num_classes)
        # print("Class map :", class_map)

        resized_class_mask = cv2.resize(
            class_mask, (224, 224), interpolation=cv2.INTER_NEAREST
        )
        resized_class_mask = np.expand_dims(resized_class_mask, axis=-1)
        categorical_mask = to_categorical(resized_class_mask, num_classes=num_classes)

        # print(lblnames[k])
        # print(class_map)
        # print(categorical_mask)
        # print(categorical_mask.shape)

        image = np.transpose(image, (2, 0, 1))
        categorical_mask = np.transpose(categorical_mask, (2, 0, 1))

        if j < SAMPLE_NUM_PER_CLASS:
            support_images[j] = image
            support_labels[j] = categorical_mask
        else:
            query_images[j - SAMPLE_NUM_PER_CLASS] = image
            query_labels[j - SAMPLE_NUM_PER_CLASS] = categorical_mask
        j += 1

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
    # init neural networks
    print("init neural networks")

    if METHOD == "multi":
        input_channel = 7
        output_channel = 4
    else:
        input_channel = 4
        output_channel = 1

    feature_encoder = CNNEncoder(input_channel)
    relation_network = RelationNetwork(output_channel)
    relation_network.apply(weights_init)
    feature_encoder_optim = torch.optim.Adam(
        feature_encoder.parameters(), lr=LEARNING_RATE
    )
    relation_network_optim = torch.optim.Adam(
        relation_network.parameters(), lr=LEARNING_RATE
    )

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    # fine-tuning
    if args.finetune:
        if os.path.exists(FEATURE_MODEL):
            feature_encoder.load_state_dict(torch.load(FEATURE_MODEL))
            feature_encoder_optim.load_state_dict(torch.load(FEATURE_OPTIM))
            print("load feature encoder success")
        else:
            print("Can not load feature encoder: %s" % FEATURE_MODEL)
            print("starting from scratch")
            feature_encoder_optim = torch.optim.Adam(
                feature_encoder.parameters(), lr=LEARNING_RATE
            )

        if os.path.exists(RELATION_MODEL):
            relation_network.load_state_dict(torch.load(RELATION_MODEL))
            relation_network_optim.load_state_dict(torch.load(RELATION_OPTIM))
            print("load relation network success")
        else:
            print("Can not load relation network: %s" % RELATION_MODEL)
            print("starting from scratch")
            relation_network_optim = torch.optim.Adam(
                relation_network.parameters(), lr=LEARNING_RATE
            )

    feature_encoder_scheduler = StepLR(
        feature_encoder_optim, step_size=EPISODE // 10, gamma=0.5
    )

    relation_network_scheduler = StepLR(
        relation_network_optim, step_size=EPISODE // 10, gamma=0.5
    )

    classes_name = os.listdir(TRAIN_DIR + "label/")  # ["o","p","z"]
    classes_index = list(range(0, len(classes_name)))  # [0, 1, 2]

    tuple_o = ()
    tuple_p = ()
    tuple_z = ()
    tuples_dic = {}

    tuple_multi = ()

    if METHOD == "multi":
        tuple_multi = load_multi_data(num=TOTAL_IMAGES_PER_CLASS)

    else:  # METHOD = iou or Random
        tuple_o, tuple_p, tuple_z = load_data(num=TOTAL_IMAGES_PER_CLASS, method=METHOD)
        tuples_dic["o"] = tuple_o
        tuples_dic["p"] = tuple_p
        tuples_dic["z"] = tuple_z

    print("Training...")

    for episode in range(args.start_episode, EPISODE):

        if METHOD == "multi":
            (samples, sample_labels, batches, batch_labels) = get_oneshot_multi_batch(
                tuple_multi
            )
            chosen_class_name = "multi"
            loss_fn = CategoricalCrossEntropyLoss().cuda(GPU)
        else:
            chosen_class_index = random.sample(
                classes_index, CLASS_NUM
            )  # it is one of [0,1,2]
            chosen_class_name = classes_name[chosen_class_index[0]]
            (samples, sample_labels, batches, batch_labels) = get_oneshot_batch(
                chosen_class_name, tuples_dic[chosen_class_name]
            )
            loss_fn = nn.BCELoss().cuda(GPU)
            # nn.MSELoss().cuda(GPU)
            # nn.BCELoss().cuda(GPU)

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

        loss = loss_fn(output, Variable(batch_labels).cuda(GPU))

        # training
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        feature_encoder_scheduler.step()
        relation_network_scheduler.step()

        if (episode + 1) % 10 == 0 and episode != EPISODE - 1:
            print(
                "episode:",
                episode + 1,
                "class:",
                chosen_class_name,
                "loss:",
                loss.cpu().data.numpy(),
            )

        elif episode == EPISODE - 1:
            print(
                "episode:",
                episode + 1,
                "class:",
                chosen_class_name,
                "loss:",
                loss.cpu().data.numpy(),
            )

        if not os.path.exists(args.TrainResultPath):
            os.makedirs(args.TrainResultPath)
        if not os.path.exists(args.ModelSavePath):
            os.makedirs(args.ModelSavePath)

        # training result visualization
        if (episode + 1) % args.ResultSaveFreq == 0:
            if METHOD == "multi":
                query_output = train_multi_visualization(
                    DISPLAY_QUERY, BATCH_NUM_PER_CLASS, output, batches, batch_labels
                )
            else:
                query_output = train_binary_visualization(
                    DISPLAY_QUERY, BATCH_NUM_PER_CLASS, output, batches, batch_labels
                )

            cv2.imwrite(
                "%s/%s_query_%s.png"
                % (args.TrainResultPath, episode + 1, chosen_class_name),
                query_output,
            )

        # save models
        if (episode + 1) % SAVE_FREQ == 0:
            torch.save(
                feature_encoder.state_dict(),
                str(
                    "./%s/%s" % (args.ModelSavePath, str(episode + 1) + "_" + METHOD)
                    + "_FE.pkl"
                ),
            )

            torch.save(
                relation_network.state_dict(),
                str(
                    "./%s/%s" % (args.ModelSavePath, str(episode + 1) + "_" + METHOD)
                    + "_RN.pkl"
                ),
            )

            torch.save(
                feature_encoder_optim.state_dict(),
                str("./trained-models/optimizer/binary_FE_optimizer.pkl"),
            )

            torch.save(
                relation_network_optim.state_dict(),
                str("./trained-models/optimizer/binary_RN_optimizer.pkl"),
            )

            print("save networks for episode:", episode + 1)


if __name__ == "__main__":
    # reset_result()
    if not LAST_GENERATE:
        clean_dataset()
        print("spiliting train and test dataset...")
        generate_dataset_folder_structure()
        print("dataset got generated successfully!!")
    main()
