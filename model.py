import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Feature Extractor Network Model
    n denotes the number frames as input. The input shape of model is adjusted accordingly.
    """
    def __init__(self, n):
        super(FeatureExtractor, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout2d(p=0.5)
        
        self.conv1 = nn.Conv2d(3*n, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.conv_final = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    
    def forward(self, x0):
        x1 = self.activation(self.conv1(x0))
        
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x3 = self.maxpool(x3)
        
        x4 = self.activation(self.conv4(x3))
        x5 = self.activation(self.conv5(x4))
        x5 = self.maxpool(x5)
        
        x6 = self.activation(self.conv6(x5))
        x7 = self.activation(self.conv7(x6))
        x7 = self.dropout(x7)
        
        x8 = torch.cat([x7, x5], dim=1)
        x8 = self.upsample(x8)
        x8 = self.activation(self.conv8(x8))
        x9 = self.activation(self.conv9(x8))
        x9 = self.dropout(x9)
        
        x10 = torch.cat([x9, x3], dim=1)
        x10 = self.upsample(x10)
        x10 = self.activation(self.conv10(x10))
        x11 = self.activation(self.conv11(x10))
        x11 = self.dropout(x11)
        
        x12 = self.conv_final(x11)
        x12 = torch.sigmoid(x12)
        
        return x12


class SegmentationNet(nn.Module):
    """
    Segmentation Model
    """
    def __init__(self):
        super(SegmentationNet, self).__init__()
        
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout2d(p=0.5)
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x0):
        x1 = self.activation(self.conv1(x0))
        x2 = self.activation(self.conv2(x1)) 
        x2 = self.dropout(x2)
        
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3)) 
        x4 = self.dropout(x4)
        
        x5 = self.activation(self.conv5(x4))
        x6 = self.activation(self.conv6(x5)) 
        x6 = self.dropout(x6)
        
        x7 = self.conv_final(x6)
        x7 = torch.sigmoid(x7)
        return x7


class Generator(nn.Module):
    """
    Feature Extractor and Semgnation Networks combined to for Generator
    n denotes the number frames as input. The input shape of model is adjusted accordingly.
    """
    def __init__(self, n):
        super(Generator, self).__init__()
        
        self.feature_extractor = FeatureExtractor(n)
        self.segmentation_net = SegmentationNet()
        
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_4 = nn.UpsamplingNearest2d(scale_factor=4)
        
    def forward(self, inputs):
        ip_scale1, ip_scale2, ip_scale3 = inputs
        
        feature_scale1 = self.feature_extractor(ip_scale1)
        feature_scale2 = self.feature_extractor(ip_scale2)
        feature_scale3 = self.feature_extractor(ip_scale3)
        
        feature_scale2_up = self.upsample_2(feature_scale2)
        feature_scale3_up = self.upsample_4(feature_scale3)
        
        feature = torch.cat([feature_scale1, feature_scale2_up, feature_scale3_up], dim=1)
        
        segmentation = self.segmentation_net(feature)
        return segmentation, feature_scale1, feature_scale2, feature_scale3


class Discriminator(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout2d(p=0.5)
        
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        
    def forward(self, x0):
        x1 = self.dropout(self.maxpool(self.activation(self.conv1(x0))))
        x2 = self.dropout(self.maxpool(self.activation(self.conv2(x1))))        
        x3 = self.dropout(self.maxpool(self.activation(self.conv3(x2))))        
        x4 = self.dropout(self.maxpool(self.activation(self.conv4(x3))))
        x5 = self.conv_final(x4)
        x5 = torch.sigmoid(x5)
        
        return x5