import torch.nn as nn
import torch
import torch.nn.functional as F

class PMGI_V4(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMGI_V4, self).__init__()
        print("PMGI_V4")
        self.features = model
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

    # 上下采样函数，用于统一尺寸
    def _up_or_down_sample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

    # 双线性池化交互
    def hbp(self, conv1, conv2):
        X = conv1 * conv2
        # X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        # X = torch.sqrt(X + 1e-5)
        X = self.avgpool(X).view(X.size()[0], -1)
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-10)
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, x, train_flag):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        # 提取后三个stage
        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        # xl1与xl3的尺寸向xl2对齐
        xl1 = self._up_or_down_sample(xl1, xl2)
        xl3 = self._up_or_down_sample(xl3, xl2)

        # HBP-Part, 三层的特征交互
        x_branch_1 = self.hbp(xl1, xl2)
        x_branch_2 = self.hbp(xl2, xl3)
        x_branch_3 = self.hbp(xl1, xl3)

        x_concat = torch.cat([x_branch_1, x_branch_2, x_branch_3], dim=1)

        # PMG-Part
        xc1 = self.classifier1(x_branch_1)
        xc2 = self.classifier2(x_branch_2)
        xc3 = self.classifier3(x_branch_3)
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

