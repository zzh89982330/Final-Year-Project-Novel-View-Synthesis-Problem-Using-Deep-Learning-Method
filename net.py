import torch.nn.functional as F
import torch.nn as nn
import torch

class Net(nn.Module):

    def __init__(self, input_chns=219, output_chns=67):
        super(Net, self).__init__()
        # Encoder:
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_chns, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Decoder:
        self.conv5_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv7_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv7_3 = nn.Sequential(
            nn.Conv2d(64, output_chns, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        conv1 = self.conv1_1(x)
        conv1 = F.pad(conv1, (0, 1, 0, 1))
        conv1 = self.conv1_2(conv1)
        
        conv2 = self.conv2_1(conv1)
        conv2 = F.pad(conv2, (0, 1, 0, 1))
        conv2 = self.conv2_2(conv2)
        
        conv3 = self.conv3_1(conv2)
        conv3 = self.conv3_2(conv3)
        conv3 = F.pad(conv3, (0, 1, 0, 1))
        conv3 = self.conv3_3(conv3)
        
        conv4 = self.conv4_3(self.conv4_2(self.conv4_1(conv3)))
        
        conv5 = torch.cat((conv4, conv3), dim=1)
        conv5 = self.conv5_1(conv5)        
        
        conv6 = self.conv6_2(self.conv6_1(torch.cat((conv5, conv2), dim=1)))
        conv7 = self.conv7_3(self.conv7_2(self.conv7_1(torch.cat((conv6, conv1), dim=1))))
        return conv7

# The Depth Estimation Network only uses the single size kernel for whole image weighting operations
class DepthEstimatonNet(nn.Module):
    def __init__(self, input_chns, output_chns):
        super(DepthEstimatonNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_chns, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, output_chns, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class OcclusionComplesionNet(nn.Module):
    """Input is the concatenated images of the input, excluding the reference image and the image used for
    plane sweep

    output channels: 3
    """
    def __init__(self, input_chns):
        super(OcclusionComplesionNet, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(input_chns, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder:
        self.conv5_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv7_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv7_3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        conv1 = self.conv1_1(x)
        conv1 = F.pad(conv1, (0, 1, 0, 1))
        conv1 = self.conv1_2(conv1)

        conv2 = self.conv2_1(conv1)
        conv2 = F.pad(conv2, (0, 1, 0, 1))
        conv2 = self.conv2_2(conv2)

        conv3 = self.conv3_1(conv2)
        conv3 = self.conv3_2(conv3)
        conv3 = F.pad(conv3, (0, 1, 0, 1))
        conv3 = self.conv3_3(conv3)

        conv4 = self.conv4_3(self.conv4_2(self.conv4_1(conv3)))

        conv5 = torch.cat((conv4, conv3), dim=1)
        conv5 = self.conv5_1(conv5)

        conv6 = self.conv6_2(self.conv6_1(torch.cat((conv5, conv2), dim=1)))
        conv7 = self.conv7_3(self.conv7_2(self.conv7_1(torch.cat((conv6, conv1), dim=1))))
        return conv7