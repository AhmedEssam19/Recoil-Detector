import torch.nn as nn


class TransferNet(nn.Module):
    def __init__(self, arch, feature_extract=False, use_dropout=False):
        super(TransferNet, self).__init__()
        self.arch = arch
        self.use_dropout = use_dropout

        if feature_extract:
            for param in self.arch.parameters():
                param.requires_grad = False

        if 'ResNet' in str(arch.__class__):
            print(arch.__class__)
            n_features = arch.fc.in_features
            self.arch.fc = nn.Linear(in_features=n_features, out_features=500)

        elif 'MobileNet' in str(arch.__class__):
            print(arch.__class__)
            self.arch.classifier[1] = nn.Linear(in_features=1280, out_features=500)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.25)

        self.final_layer = nn.Linear(500, 1)

    def forward(self, x):
        features = self.arch(x)
        if self.use_dropout:
            features = self.dropout(features)

        return self.final_layer(features)
