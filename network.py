import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torch.autograd import Variable


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self, input_channel):
        super(CNNEncoder, self).__init__()
        features = list(models.vgg16_bn().features)  # pretrained=args.loadImagenet
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)
        )
        self.features = nn.ModuleList(features)[1:]  # .eval()
        # print (nn.Sequential(*list(models.vgg16_bn(pretrained=True).children())[0]))
        # self.features = nn.ModuleList(features).eval()
        # print(features)
        # print("**********")
        # print(self.features)

    def forward(self, x):
        results = []
        x = self.layer1(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:
                results.append(x)

        return x, results


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, output_channel):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
        )  # 14 x 14
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
        )  # 28 x 28
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(),
        )  # 56 x 56
        self.double_conv4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
        )  # 112 x 112
        self.double_conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, output_channel, kernel_size=1, padding=0),
        )  # 256 x 256

    def forward(self, x, concat_features, active_function):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.upsample(out)  # block 1
        out = torch.cat((out, concat_features[-1]), dim=1)
        out = self.double_conv1(out)
        out = self.upsample(out)  # block 2
        out = torch.cat((out, concat_features[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out)  # block 3
        out = torch.cat((out, concat_features[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out)  # block 4
        out = torch.cat((out, concat_features[-4]), dim=1)
        out = self.double_conv4(out)
        out = self.upsample(out)  # block 5
        out = torch.cat((out, concat_features[-5]), dim=1)
        out = self.double_conv5(out)

        if active_function == "sigmoid":
            out = F.sigmoid(out)

        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm") != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def calculate_output(
    samples,
    batches,
    feature_encoder,
    relation_network,
    CLASS_NUM,
    METHOD,
    SAMPLE_NUM_PER_CLASS,
    BATCH_NUM_PER_CLASS,
    GPU,
):

    if METHOD == "multi":
        output_channel = 4
        active_function = "softmax"

    else:
        output_channel = 1
        active_function = "sigmoid"

    # calculate features
    sample_features, _ = feature_encoder(Variable(samples).cuda(GPU))
    sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 512, 7, 7)
    sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
    sample_features_ext = sample_features.unsqueeze(0).repeat(
        BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1
    )
    # print("sample_features_ext = ",sample_features_ext.shape)
    # calculate relations
    batch_features, ft_list = feature_encoder(Variable(batches).cuda(GPU))
    batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

    # print("batch_features_ext= ", batch_features_ext.shape)
    # print("ft list = ", ft_list)

    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(
        -1, 1024, 7, 7
    )
    # print("relation pair", relation_pairs.shape)
    output = relation_network(relation_pairs, ft_list, active_function).view(
        -1, output_channel, 224, 224
    )

    return output
