#  Fully convolutional net that receive image and predict XYZ maps (3 layers per image) and segmentation maps (2 layers).
import torch
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from vision_service import Visualization as vis
#import Visualization as vis
######################################################################################################################
class Net(nn.Module):
######################################################################################################################
    def __init__(self, MaskList,XYZList): # MaskList is list of segmentation mask to predict, XYZList is list of XYZ map to predict

        # --------------Build layers for standart FCN with only image as input------------------------------------------------------
            super(Net, self).__init__()
            # ---------------Load pretrained  encoder---------------------------------------------------------
            self.Encoder = models.resnet101()

#---------------------------------Dilated convolution ASPP layers (same as deep lab)------------------------------------------------------------------------------

            self.ASPPScales = [1, 2, 4, 12, 16]
            self.ASPPLayers = nn.ModuleList()
            for scale in self.ASPPScales:
                    self.ASPPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 512, stride=1, kernel_size=3,  padding = (scale, scale), dilation = (scale, scale), bias=False),nn.BatchNorm2d(512),nn.ReLU()))

#-------------------------------------Squeeze ASPP Layer------------------------------------------------------------------------------
            self.SqueezeLayers = nn.Sequential(
                nn.Conv2d(2560, 512, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()#,
                # nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(512),
                # nn.ReLU()
            )
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()))
            # ------------------Skip connection squeeze applied to the (concat of upsample+skip conncection layers)-----------------------------------------------------------------------------
            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 512, 256, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 128, 256, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))

            # ----------------Final prediction XYZ maps------------------------------------------------------------------------------------------
            self.OutLayersList = nn.ModuleList()
            self.OutLayersDicXYZ={}
            self.OutLayersDicMask = {}
            for nm in  XYZList:
                        self.OutLayersDicXYZ[nm]=nn.Conv2d(256, 3, stride=1, kernel_size=3, padding=1, bias=False)
                        self.OutLayersList.append(self.OutLayersDicXYZ[nm])
        # ----------------Final prediction segmentation Mask------------------------------------------------------------------------------------------

            self.OutLayersDicMask = {}
            for nm in MaskList:
                self.OutLayersDicMask[nm] = nn.Conv2d(256, 2, stride=1, kernel_size=3, padding=1, bias=False)
                self.OutLayersList.append(self.OutLayersDicMask[nm])

##########################################################################################################################################################
    def forward(self, Images,  UseGPU=True, TrainMode=True,PredictXYZ=True,PredictMasks=True, FreezeBatchNorm_EvalON=False):

               # ----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]

                if TrainMode == True:
                   tp = torch.FloatTensor # Training mode
                else:
                   tp = torch.half
                   #      self.eval()
                   self.half()
                if FreezeBatchNorm_EvalON: self.eval() # dont Update batch nor mstatiticls

                # Convert input to pytorch
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(np.float32)), requires_grad=False).transpose(2,3).transpose(1, 2).type(tp)

# ---------------Convert to cuda gpu-------------------------------------------------------------------------------------------------------------------

                if UseGPU:
                    InpImages = InpImages.cuda()
                    self.cuda()
                else:
                    InpImages = InpImages.cpu().float()
                    self.cpu().float()
#----------------Normalize image values-----------------------------------------------------------------------------------------------------------
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
#---------------Run Encoder-----------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                EncoderMap = self.Encoder.layer4(x)

#---------------------------------ASPP Layers (Dilated conv)--------------------------------------------------------------------------------
                ASPPFeatures = []  # Results of various of scaled procceessing
                for ASPPLayer in self.ASPPLayers:
                    y = ASPPLayer( EncoderMap )
                    ASPPFeatures.append(y)
                x = torch.cat(ASPPFeatures, dim=1)
                x = self.SqueezeLayers(x)
#----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear',align_corners=False)  # Upsample
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1) # Apply skip connection and concat with upsample
                  x = self.SqueezeUpsample[i](x) # Squeeze

    # ---------------------------------Final XYZ map prediction-------------------------------------------------------------------------------
                self.OutXYZ = {}
                if PredictXYZ:
                    for nm in self.OutLayersDicXYZ:
                        # print(nm)
                        l = self.OutLayersDicXYZ[nm](x)
                        if TrainMode == False:  # For prediction mode resize to the input image size
                                     l = nn.functional.interpolate(l, size=InpImages.shape[2:4], mode='bilinear',align_corners=False)  # Resize to original image size
                        self.OutXYZ[nm]=l
    #--------------------------Output segmentation mask---------------------------------------------------------------------------------------
                self.OutProbMask = {}
                self.OutMask = {}
                if PredictMasks:
                    for nm in self.OutLayersDicMask:
                        l=self.OutLayersDicMask[nm](x)
                        if TrainMode==False: # For prediction mode resize to the input image size
                                 l = nn.functional.interpolate(l, size=InpImages.shape[2:4], mode='bilinear',align_corners=False)  # Resize to original image size
                        Prob = F.softmax(l, dim=1)  # Calculate class probability per pixel
                        tt, Labels = l.max(1)  # Find label per pixel
                        self.OutProbMask[nm]=Prob
                        self.OutMask[nm]=Labels
                return self.OutXYZ, self.OutProbMask, self.OutMask

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
