import torch.nn as nn
import torch


class PMGI_V3_Extend(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMGI_V3_Extend, self).__init__()

        self.features = model
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.map1 = nn.Linear((self.num_ftrs // 2) * 3, feature_size)
        self.map2 = nn.Linear(feature_size, (self.num_ftrs // 2))
        self.fc = nn.Linear(self.num_ftrs // 2, classes_num)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, train_flag):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        # PMG-Part
        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xl1 = self.maxpool(xl1)
        xl1 = xl1.view(xl1.size(0), -1)

        xl2 = self.maxpool(xl2)
        xl2 = xl2.view(xl2.size(0), -1)

        xl3 = self.maxpool(xl3)
        xl3 = xl3.view(xl3.size(0), -1)

        x_concat = torch.cat((xl1, xl2, xl3), -1)

        if train_flag == "train":
            # API-Part
            feas = self.map1(x_concat)
            feas = self.drop(feas)
            feas = self.map2(feas)

            gate1 = torch.mul(feas, xl1)
            gate1 = self.sigmoid(gate1)
            gate2 = torch.mul(feas, xl2)
            gate2 = self.sigmoid(gate2)
            gate3 = torch.mul(feas, xl3)
            gate3 = self.sigmoid(gate3)

            x1 = torch.mul(gate1, xl1) + xl1
            x2 = torch.mul(gate2, xl2) + xl2
            x3 = torch.mul(gate3, xl3) + xl3


            xc1 = self.classifier1(x1)
            xc2 = self.classifier2(x2)
            xc3 = self.classifier3(x3)
            # or
            # xc1 = self.classifier1(x1)
            # xc2 = self.classifier1(x2)
            # xc3 = self.classifier1(x3)
            # or
            # xc1 = self.fc(x1)
            # xc2 = self.fc(x2)
            # xc3 = self.fc(x3)

            features = torch.cat([x1, x2, x3], dim=1)
            x_concat = self.classifier_concat(features)

        if train_flag == "val":
            xc1 = self.classifier1(xl1)
            xc2 = self.classifier2(xl2)
            xc3 = self.classifier3(xl3)
            x_concat = self.classifier_concat(x_concat)

        return xc1, xc2, xc3, x_concat


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
