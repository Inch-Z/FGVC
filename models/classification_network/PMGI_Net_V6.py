import torch.nn as nn
import torch


class PMGI_V6(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMGI_V6, self).__init__()
        print("PMGI_V6")
        self.features = model
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(feature_size * 3),
            nn.ELU(inplace=True),
            nn.Linear(feature_size * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, feature_size, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, feature_size, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, feature_size, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.map1 = nn.Linear(feature_size * 3, feature_size)
        self.map2 = nn.Linear(feature_size, feature_size)
        self.fc = nn.Linear(feature_size // 2, classes_num)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2, x3, train_flag):
        _, _, _, _, x1 = self.features(x1)
        _, _, _, _, x2 = self.features(x2)
        _, _, _, _, x3 = self.features(x3)

        x1 = self.conv_block1(x1)  # [bs, feature-size, 14, 14]
        x2 = self.conv_block1(x2)  # [bs, feature-size, 14, 14]
        x3 = self.conv_block1(x3)  # [bs, feature-size, 14, 14]

        xl1 = self.maxpool(x1)
        xl1 = xl1.view(xl1.size(0), -1)

        xl2 = self.maxpool(x2)
        xl2 = xl2.view(xl2.size(0), -1)

        xl3 = self.maxpool(x3)
        xl3 = xl3.view(xl3.size(0), -1)

        x_concat = torch.cat([xl1, xl2, xl3], dim=1)

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


        # PMG-Part
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
