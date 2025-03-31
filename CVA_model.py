import torch
import torch.nn as nn

class Feature_encoder(nn.Module):
    def __init__(self):
        super(Feature_encoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, height, width)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        return x


class Attribute_encoder(nn.Module):
    def __init__(self):
        super(Attribute_encoder, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 11)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Feature_encoder = Feature_encoder()
        self.Attribute_encoder = Attribute_encoder()

        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(11, 32)
        self.fc3 = nn.Linear(32 * 2, 10)
        self.relu = nn.LeakyReLU()

    def forward_once(self, x):
        output = self.Feature_encoder(x)
        output = torch.flatten(output, 1)
        return output

    def forward(self, input1):
        output_feature = self.forward_once(input1)
        output_attribute = self.Attribute_encoder(output_feature)
        x1 = self.relu(self.fc1(output_feature))
        x2 = self.relu(self.fc2(output_attribute))
        x = torch.cat((x1, x2), dim=1)
        class_logits = self.fc3(x)

        return output_feature, output_attribute, class_logits
