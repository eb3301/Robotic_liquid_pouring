import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Troppo semplice, underfit
class VolumeNet(nn.Module):
    # Modello troppo semplice --> underfit
    def __init__(self, input_channels=2,size=(256, 192)):
        super(VolumeNet, self).__init__()
        self.Im_size=list(size)
        pool_size=3
        self.Im_size[0]//=2**pool_size
        self.Im_size[1]//=2**pool_size
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * self.Im_size[0] * self.Im_size[1], 128) # size last conv * sizes after pooling
        self.fc2 = nn.Linear(128, 1)  # regression

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 640,480 → 320,240
        x = self.pool(F.relu(self.conv2(x)))  # 320,240 → 160,120
        x = self.pool(F.relu(self.conv3(x)))  # 160,120 → 80,60
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # output: pred vol
        return x

class VolumeNN(nn.Module):
    def __init__(self, input_channels=2,dropout_rate=0.2,size=(256, 192)):
        super(VolumeNN, self).__init__()

        self.Im_size=list(size)
        pool_size=5
        self.Im_size[0]//=2**pool_size
        self.Im_size[1]//=2**pool_size

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=64 * self.Im_size[0] * self.Im_size[1], out_features=1024)  # size last conv * sizes after pooling
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2) # 320,240 → 160,120
        x = F.dropout2d(x, p=self.dropout_rate)
 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.max_pool2d(x, kernel_size=2) # 640,480 → 320,240
        x = F.dropout2d(x, p=self.dropout_rate)

        x = F.relu(self.bn3(self.conv3(x))) 
        x = F.max_pool2d(x, kernel_size=2) # 160,120 → 80,60
        x = F.dropout2d(x, p=self.dropout_rate)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2) # 80,60 → 40,30
        x = F.dropout2d(x, p=self.dropout_rate)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=2) # 40,30 → 20,15
        x = F.dropout2d(x, p=self.dropout_rate)

        x = x.view(-1, 64 * self.Im_size[0] * self.Im_size[1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.fc4(x)

        return x

class VolumeNetPret(nn.Module):
    def __init__(self, backbone_name="ResNet18", input_channels=4, dropout_p=0.3, pretrained=True):
        """
        Modello generico per regressione del volume
        Args:
            backbone_name (str): "ResNet18", "ResNet34", "ResNet50", "EfficientNet_b0", ...
            input_channels (int): numero di canali in input (default 4)
            dropout_p (float): dropout prima del layer finale
            pretrained (bool): se True, carica pesi ImageNet
        """
        super().__init__()

        # Carica backbone
        if "ResNet" in backbone_name:
            weights = getattr(models, f"{backbone_name}_Weights").IMAGENET1K_V1 if pretrained else None
            self.backbone = getattr(models, backbone_name.lower())(weights=weights)

            # Adatta conv1 per input_channels
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )

            if pretrained and input_channels > 3:
                with torch.no_grad():
                    self.backbone.conv1.weight[:, :3, :, :] = old_conv.weight # primi tre canali hanno pesi vecchi
                    self.backbone.conv1.weight[:, 3:input_channels, :, :] = old_conv.weight.mean(dim=1, keepdim=True) # ultimo canale ha media pesi

            # Replace FC layer con regressore
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, 1)
            )

        elif "EfficientNet" in backbone_name:
            weights = getattr(models, f"{backbone_name.upper()}_Weights").IMAGENET1K_V1 if pretrained else None
            self.backbone = getattr(models, backbone_name.lower())(weights=weights)

            # Adatta primo conv per input_channels
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )

            if pretrained and input_channels > 3:
                with torch.no_grad():
                    self.backbone.features[0][0].weight[:, :3, :, :] = old_conv.weight
                    self.backbone.features[0][0].weight[:, 3:input_channels, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

            # Replace classifier per regressione
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, 1)
            )

        else:
            raise ValueError(f"Backbone {backbone_name} non supportato")

    def forward(self, x):
        return self.backbone(x)
