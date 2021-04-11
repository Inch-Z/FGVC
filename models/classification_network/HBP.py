import torch
import torchvision




class HBP(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())
        [:-5])
        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())
        [-5:-3])
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())
        [-3:-1])
        self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512, 8192, kernel_size=1, bias=False),
                                                 torch.nn.BatchNorm2d(8192),
                                                 torch.nn.ReLU(inplace=True))
        # Linear classifier.
        self.fc = torch.nn.Linear(8192 * 3, 200)

        # Freeze all previous layers.
        for param in self.features_conv5_1.parameters():
            param.requires_grad = False
        for param in self.features_conv5_2.parameters():
            param.requires_grad = False
        for param in self.features_conv5_3.parameters():
            param.requires_grad = False

        # Initialize the fc layers.
        torch.nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

        # init
        for m in self.bilinear_proj.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def hbp(self, conv1, conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)
        assert (proj_1.size() == (N, 8192, 28, 28))
        X = proj_1 * proj_2
        assert (X.size() == (N, 8192, 28, 28))
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, X):
        N = X.size()[0]
        # assert X.size() == (N, 3, 448, 448)
        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)

        X_branch_1 = self.hbp(X_conv5_1, X_conv5_2)
        X_branch_2 = self.hbp(X_conv5_2, X_conv5_3)
        X_branch_3 = self.hbp(X_conv5_1, X_conv5_3)

        X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)
        assert X_branch.size() == (N, 8192 * 3)
        X = self.fc(X_branch)
        assert X.size() == (N, 200)
        return X
