import torch
import torch.nn as nn
import torchvision.models as models

class SDEModel(nn.Module):
    def __init__(self, n_classes=3):
        super(SDEModel, self).__init__()
        self.deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.deeplab.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)

        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()  # Normalizza in [0,1]
        )

    def forward(self, x):
        features = self.deeplab.backbone(x)['out']
        seg_logits = self.deeplab.classifier(features)
        depth_map = self.depth_head(features)
        return seg_logits, depth_map
